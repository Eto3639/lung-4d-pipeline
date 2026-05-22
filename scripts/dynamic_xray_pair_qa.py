"""Pair AP and lateral dynamic X-ray recordings per case.

For each ``CaseNN/`` directory under the anonymized DICOM root, treat the
first DICOM (``rec1``) as the AP view and the second (``rec2``) as the
lateral view — the Konica DDR SeriesDescription field reads ``正面 A-P``
for both, so we rely on the recording index rather than that tag. (Override
with ``--ap-index`` / ``--lat-index`` if your convention differs.)

For each case the pipeline:
  1. loads both multi-frame DICOMs into ``(T, H, W)`` numpy stacks,
  2. extracts a respiratory signal per view via Amsterdam Shroud,
  3. picks the AP/LAT one-breath pair with the highest length-normalized NCC,
  4. saves the paired frame slices as ``.npy`` (downstream 2D-4D model input),
  5. writes a per-case PDF + HTML + metrics.json,
  6. aggregates into ``index.html`` + ``summary.json``.

Patient-derived outputs stay off the public repo via ``.gitignore``.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html as html_mod
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.backends.backend_pdf import PdfPages

from dvf_qa.amsterdam_shroud import respiratory_signal
from dvf_qa.cycle_pairing import (
    detect_cycle_boundaries,
    pair_best_correlated_cycles,
)


def load_frames_from_dicom(path: Path) -> tuple[np.ndarray, float, dict]:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    slope = float(ds.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(ds.get("RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    photometric = str(ds.get("PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr
    frame_time_ms = float(ds.get("FrameTime", 0.0) or 0.0)
    fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 15.0
    meta = {
        "n_frames": int(arr.shape[0]),
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
        "fps": float(fps),
        "frame_time_ms": float(frame_time_ms),
        "photometric_interpretation": photometric,
        "series_description": str(ds.get("SeriesDescription", "")),
        "anon_id": str(ds.get("PatientID", "")),
    }
    return arr, fps, meta


# ---------- figures ----------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fig_mean_frames(ap: np.ndarray, lat: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(ap.mean(axis=0), cmap="gray"); axes[0].set_title("AP mean")
    axes[1].imshow(lat.mean(axis=0), cmap="gray"); axes[1].set_title("Lateral mean")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    return fig


def fig_signals(ap_res, lat_res, ap_fps, lat_fps, ap_boundaries, lat_boundaries, pair):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, res, fps, boundaries, view, sel in [
        (axes[0], ap_res, ap_fps, ap_boundaries, "AP", (pair.ap_start, pair.ap_end)),
        (axes[1], lat_res, lat_fps, lat_boundaries, "Lateral", (pair.lateral_start, pair.lateral_end)),
    ]:
        t = np.arange(res.signal.size) / fps
        ax.plot(t, res.signal, color="tab:gray", lw=0.8)
        for b in boundaries:
            if 0 <= b < t.size:
                ax.axvline(t[b], color="tab:red", alpha=0.4, lw=0.6, linestyle="--")
        ax.axvspan(sel[0] / fps, sel[1] / fps,
                   color="tab:blue" if view == "AP" else "tab:orange", alpha=0.2,
                   label="selected cycle")
        ax.set_title(f"{view} — f={res.dominant_frequency_hz:.3f} Hz, "
                     f"period={res.period_frames:.1f} fr, "
                     f"{len(boundaries)} boundaries")
        ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3); ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_pair_overlay(pair):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0.0, 1.0, pair.resample_length)
    ax.plot(x, pair.ap_resampled, label="AP (z-scored)", color="tab:blue")
    ax.plot(x, pair.lateral_resampled, label="Lateral (z-scored)", color="tab:orange")
    ax.set_xlabel("Cycle fraction"); ax.set_ylabel("normalized signal")
    ax.set_title(f"Best paired cycle  |  NCC = {pair.correlation:.3f}")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def fig_pair_matrix(pair):
    pairs = pair.all_pairs
    ap_starts = sorted({p["ap_start"] for p in pairs})
    lat_starts = sorted({p["lateral_start"] for p in pairs})
    if not ap_starts or not lat_starts:
        fig, ax = plt.subplots(figsize=(5, 2)); ax.axis("off")
        ax.text(0.5, 0.5, "(no pair matrix)", ha="center", va="center")
        return fig
    M = np.full((len(ap_starts), len(lat_starts)), np.nan)
    for p in pairs:
        M[ap_starts.index(p["ap_start"]), lat_starts.index(p["lateral_start"])] = p["correlation"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xlabel("Lateral cycle index"); ax.set_ylabel("AP cycle index")
    ax.set_title(f"NCC matrix (best={pair.correlation:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def fig_paired_frames(ap: np.ndarray, lat: np.ndarray, pair):
    """Show 3 frames each from AP and lateral selected cycles (start / mid / end)."""
    ap_idxs = [pair.ap_start, (pair.ap_start + pair.ap_end) // 2, max(pair.ap_end - 1, pair.ap_start)]
    lat_idxs = [pair.lateral_start, (pair.lateral_start + pair.lateral_end) // 2,
                max(pair.lateral_end - 1, pair.lateral_start)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for c, t in enumerate(ap_idxs):
        axes[0, c].imshow(ap[t], cmap="gray")
        axes[0, c].set_title(f"AP frame {t}")
    for c, t in enumerate(lat_idxs):
        axes[1, c].imshow(lat[t], cmap="gray")
        axes[1, c].set_title(f"Lateral frame {t}")
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


# ---------- per-case ----------

def run_case(case_dir: Path, out_dir: Path, *, ap_idx: int, lat_idx: int) -> dict:
    dcm_files = sorted(p for p in case_dir.iterdir() if p.suffix == ".dcm")
    if len(dcm_files) < max(ap_idx, lat_idx) + 1:
        raise ValueError(f"need at least {max(ap_idx, lat_idx) + 1} DICOMs, got {len(dcm_files)}")
    ap_path = dcm_files[ap_idx]
    lat_path = dcm_files[lat_idx]
    case_id = case_dir.name
    print(f"\n=== case {case_id} ===")
    print(f"  AP : {ap_path.name}")
    print(f"  Lat: {lat_path.name}")

    ap_frames, ap_fps, ap_meta = load_frames_from_dicom(ap_path)
    lat_frames, lat_fps, lat_meta = load_frames_from_dicom(lat_path)
    print(f"  AP  {ap_meta['n_frames']} fr @ {ap_fps:.2f} fps, "
          f"Lat {lat_meta['n_frames']} fr @ {lat_fps:.2f} fps")

    ap_res = respiratory_signal(ap_frames, fps=ap_fps)
    lat_res = respiratory_signal(lat_frames, fps=lat_fps)
    ap_b = detect_cycle_boundaries(ap_res.signal, ap_fps, min_period_s=2.0)
    lat_b = detect_cycle_boundaries(lat_res.signal, lat_fps, min_period_s=2.0)
    print(f"  AP  f={ap_res.dominant_frequency_hz:.3f} Hz, {len(ap_b)} boundaries")
    print(f"  Lat f={lat_res.dominant_frequency_hz:.3f} Hz, {len(lat_b)} boundaries")

    pair = pair_best_correlated_cycles(
        ap_res.signal, lat_res.signal,
        ap_fps=ap_fps, lateral_fps=lat_fps,
        min_period_s=2.0,
    )
    print(f"  NCC = {pair.correlation:.3f}  AP[{pair.ap_start}:{pair.ap_end}]  "
          f"Lat[{pair.lateral_start}:{pair.lateral_end}]")

    case_out = out_dir / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "case_id": case_id,
        "ap_anon_id": ap_meta["anon_id"],
        "lateral_anon_id": lat_meta["anon_id"],
        "ap_n_frames": ap_meta["n_frames"],
        "lateral_n_frames": lat_meta["n_frames"],
        "ap_fps": ap_fps,
        "lateral_fps": lat_fps,
        "ap_dominant_frequency_hz": float(ap_res.dominant_frequency_hz),
        "lateral_dominant_frequency_hz": float(lat_res.dominant_frequency_hz),
        "ap_period_frames": float(ap_res.period_frames),
        "lateral_period_frames": float(lat_res.period_frames),
        "ap_cycle_boundary_count": int(len(ap_b)),
        "lateral_cycle_boundary_count": int(len(lat_b)),
        "best_pair_ncc": float(pair.correlation),
        "ap_cycle_frames": [int(pair.ap_start), int(pair.ap_end)],
        "lateral_cycle_frames": [int(pair.lateral_start), int(pair.lateral_end)],
        "ap_cycle_duration_s": (pair.ap_end - pair.ap_start) / ap_fps,
        "lateral_cycle_duration_s": (pair.lateral_end - pair.lateral_start) / lat_fps,
    }
    (case_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Save paired frame slices for downstream 2D-4D model input
    np.save(case_out / "ap_cycle_frames.npy", ap_frames[pair.ap_start:pair.ap_end])
    np.save(case_out / "lateral_cycle_frames.npy", lat_frames[pair.lateral_start:pair.lateral_end])

    figs = {
        "Mean frames (AP / Lateral)": fig_mean_frames(ap_frames, lat_frames),
        "Respiratory signals + boundaries": fig_signals(ap_res, lat_res, ap_fps, lat_fps, ap_b, lat_b, pair),
        "Selected pair overlay (z-scored)": fig_pair_overlay(pair),
        "NCC matrix": fig_pair_matrix(pair),
        "Paired cycle key frames": fig_paired_frames(ap_frames, lat_frames, pair),
    }
    write_pdf(case_out / "report.pdf", case_id, metrics, figs)
    figs_for_html = {
        "Mean frames (AP / Lateral)": fig_mean_frames(ap_frames, lat_frames),
        "Respiratory signals + boundaries": fig_signals(ap_res, lat_res, ap_fps, lat_fps, ap_b, lat_b, pair),
        "Selected pair overlay (z-scored)": fig_pair_overlay(pair),
        "NCC matrix": fig_pair_matrix(pair),
        "Paired cycle key frames": fig_paired_frames(ap_frames, lat_frames, pair),
    }
    write_html(case_out / "report.html", case_id, metrics, figs_for_html)
    return metrics


def write_pdf(path: Path, case_id: str, metrics: dict, figs: dict) -> None:
    with PdfPages(path) as pdf:
        # cover with metrics summary
        fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis("off")
        ax.text(0.5, 0.97, f"Dynamic chest X-ray pair QA — {case_id}",
                ha="center", va="top", fontsize=18, fontweight="bold")
        ap_dur = metrics["ap_n_frames"] / metrics["ap_fps"]
        lat_dur = metrics["lateral_n_frames"] / metrics["lateral_fps"]
        rows = [
            ("AP recording",             f"{metrics['ap_n_frames']} fr @ {metrics['ap_fps']:.2f} fps ({ap_dur:.2f}s)"),
            ("Lateral recording",        f"{metrics['lateral_n_frames']} fr @ {metrics['lateral_fps']:.2f} fps ({lat_dur:.2f}s)"),
            ("AP f (Hz)",                f"{metrics['ap_dominant_frequency_hz']:.4f}"),
            ("Lateral f (Hz)",           f"{metrics['lateral_dominant_frequency_hz']:.4f}"),
            ("AP cycle boundaries",      str(metrics['ap_cycle_boundary_count'])),
            ("Lateral cycle boundaries", str(metrics['lateral_cycle_boundary_count'])),
            ("Best pair NCC",            f"{metrics['best_pair_ncc']:.4f}"),
            ("AP cycle frames",          f"{metrics['ap_cycle_frames'][0]} - {metrics['ap_cycle_frames'][1]} "
                                          f"({metrics['ap_cycle_duration_s']:.2f}s)"),
            ("Lateral cycle frames",     f"{metrics['lateral_cycle_frames'][0]} - {metrics['lateral_cycle_frames'][1]} "
                                          f"({metrics['lateral_cycle_duration_s']:.2f}s)"),
        ]
        table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                         cellLoc="left", colLoc="left", loc="center",
                         colWidths=[0.45, 0.5])
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        for title, fig in figs.items():
            fig.suptitle(title, fontsize=12, y=1.02)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_html(path: Path, case_id: str, metrics: dict, figs: dict) -> None:
    sections = [
        f"<header><h1>Dynamic chest X-ray pair QA — {html_mod.escape(case_id)}</h1>"
        f"<p>AP {metrics['ap_n_frames']} fr / Lateral {metrics['lateral_n_frames']} fr "
        f"@ {metrics['ap_fps']:.2f} fps · "
        f"AP f=<b>{metrics['ap_dominant_frequency_hz']:.3f} Hz</b>, "
        f"Lateral f=<b>{metrics['lateral_dominant_frequency_hz']:.3f} Hz</b><br/>"
        f"Best pair <b>NCC = {metrics['best_pair_ncc']:.4f}</b> · "
        f"AP cycle frames <b>{metrics['ap_cycle_frames'][0]}-{metrics['ap_cycle_frames'][1]}</b> "
        f"({metrics['ap_cycle_duration_s']:.2f}s), "
        f"Lateral <b>{metrics['lateral_cycle_frames'][0]}-{metrics['lateral_cycle_frames'][1]}</b> "
        f"({metrics['lateral_cycle_duration_s']:.2f}s)</p></header>"
    ]
    for title, fig in figs.items():
        b64 = _fig_to_b64(fig)
        sections.append(f"<section><h2>{html_mod.escape(title)}</h2>"
                        f"<img src='data:image/png;base64,{b64}'/></section>")
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<title>Dynamic chest X-ray pair QA — {html_mod.escape(case_id)}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-top: 16px; }} h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
img {{ max-width: 100%; height: auto; }}
section {{ margin-bottom: 16px; }}
</style></head><body>
{chr(10).join(sections)}
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_top_index(out_dir: Path, results: list[dict]) -> None:
    rows = []
    for r in sorted(results, key=lambda x: x["case_id"]):
        cid = r["case_id"]
        ncc = r["best_pair_ncc"]
        cls = "good" if ncc >= 0.9 else ("warn" if ncc >= 0.5 else "fail")
        rows.append(
            f"<tr><td><a href='{cid}/report.html'>{html_mod.escape(cid)}</a></td>"
            f"<td><a href='{cid}/report.pdf'>PDF</a></td>"
            f"<td>{r['ap_dominant_frequency_hz']:.3f}</td>"
            f"<td>{r['lateral_dominant_frequency_hz']:.3f}</td>"
            f"<td class='{cls}'>{ncc:.3f}</td>"
            f"<td>{r['ap_cycle_frames'][0]}-{r['ap_cycle_frames'][1]}</td>"
            f"<td>{r['lateral_cycle_frames'][0]}-{r['lateral_cycle_frames'][1]}</td>"
            f"<td>{r['ap_cycle_duration_s']:.2f}s / {r['lateral_cycle_duration_s']:.2f}s</td></tr>"
        )
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>Dynamic chest X-ray pair QA</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
th {{ background: #eee; }}
.good {{ background: #d4f5d4; }} .warn {{ background: #fff3cd; }} .fail {{ background: #fbd1d1; }}
</style></head>
<body>
<h1>Dynamic chest X-ray — AP/Lateral pair QA</h1>
<p>Per-case Amsterdam Shroud signals for AP (rec1) and Lateral (rec2),
followed by cycle pairing to extract the highest-NCC one-breath pair.</p>
<table>
<tr><th>Case</th><th>PDF</th><th>AP f (Hz)</th><th>Lat f (Hz)</th>
<th>NCC</th><th>AP cycle</th><th>Lat cycle</th><th>Duration AP/Lat</th></tr>
{chr(10).join(rows)}
</table>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Anonymized DICOM root (CaseNN/*.dcm)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ap-index", type=int, default=0,
                        help="0-based index of the AP recording within each case dir (default 0=rec1)")
    parser.add_argument("--lat-index", type=int, default=1,
                        help="0-based index of the Lateral recording (default 1=rec2)")
    parser.add_argument("--cases", nargs="*",
                        help="Optional subset of case directory names")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    if args.cases:
        chosen = set(args.cases)
        case_dirs = [d for d in case_dirs if d.name in chosen]
    if not case_dirs:
        print(f"No case dirs under {args.root}", file=sys.stderr); return 2

    results: list[dict] = []
    for case_dir in case_dirs:
        try:
            results.append(run_case(case_dir, args.out_dir,
                                    ap_idx=args.ap_index, lat_idx=args.lat_index))
        except Exception:
            print(f"FAILED case {case_dir.name}:")
            traceback.print_exc()
    if results:
        write_top_index(args.out_dir, results)
        print(f"\nIndex: {args.out_dir / 'index.html'}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
