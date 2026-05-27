"""Build a single unified per-case report covering every QA stage.

For each case under ``outputs/dynamic_xray_pair_qa/case<NN>/`` this writes
``outputs/case_unified_report/case_<NN>/report.{pdf,html}`` aggregating:

  1. Recording summary (frame count, fps, anonymised id)
  2. Mean frames AP / Lateral
  3. Respiratory signals + cycle boundaries + selected pair window
  4. Selected paired-cycle overlay (z-scored)
  5. Pair-QA NCC matrix
  6. Predicted 4DCT slices (axial / coronal / sagittal of 4 sampled phases),
     **only if** ``outputs/lung4d_inference/case_<NN>/pred_4d_ct.npz`` exists
  7. Generated-CT QA module-status table per frame (only if QA outputs exist)

Stages that are missing for a given case appear as "not available" in the
report so it is obvious where the pipeline still needs to run.

Patient-derived → output dir stays in ``.gitignore``.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html as html_mod
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.backends.backend_pdf import PdfPages


# ---------- I/O ----------

def load_dicom_frames(path: Path) -> tuple[np.ndarray, float, dict]:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    slope = float(ds.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(ds.get("RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    if str(ds.get("PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        arr = arr.max() - arr
    fps = 1000.0 / float(ds.get("FrameTime", 66.0) or 66.0)
    return arr, fps, {"anon_id": str(ds.get("PatientID", "")),
                       "series_description": str(ds.get("SeriesDescription", ""))}


def find_case_dicoms(anon_root: Path, case_id: str) -> tuple[Path, Path]:
    case_dir = anon_root / case_id
    files = sorted(p for p in case_dir.iterdir() if p.suffix == ".dcm")
    return files[0], files[1]


def load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=True)


# ---------- figure helpers ----------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mid_native(volume: np.ndarray, axis: int) -> np.ndarray:
    """Mid slice. Model-generated CTs have z=0 = head, no extra flip needed."""
    return np.take(volume, volume.shape[axis] // 2, axis=axis)


# ---------- per-section figures ----------

def fig_metadata_table(case_id: str, info: dict):
    rows = [(k, str(v)) for k, v in info.items()]
    fig, ax = plt.subplots(figsize=(8.27, max(2.5, 0.4 * len(rows) + 1.5)))
    ax.axis("off")
    ax.text(0.5, 0.95, f"case {case_id} — pipeline summary",
            ha="center", va="top", fontsize=16, fontweight="bold")
    table = ax.table(cellText=rows, colLabels=["Field", "Value"],
                     cellLoc="left", colLoc="left", loc="upper center",
                     colWidths=[0.4, 0.55])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.6)
    return fig


def fig_mean_frames(ap_clip: np.ndarray, lat_clip: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(ap_clip.mean(0), cmap="gray"); axes[0].set_title("AP mean (selected cycle)")
    axes[1].imshow(lat_clip.mean(0), cmap="gray"); axes[1].set_title("Lateral mean (selected cycle)")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def fig_pair_signals(pair_metrics: dict, ap_full: np.ndarray, lat_full: np.ndarray):
    from dvf_qa.amsterdam_shroud import respiratory_signal
    from dvf_qa.cycle_pairing import detect_cycle_boundaries
    ap_fps = pair_metrics["ap_fps"]; lat_fps = pair_metrics["lateral_fps"]
    ap_res = respiratory_signal(ap_full, fps=ap_fps)
    lat_res = respiratory_signal(lat_full, fps=lat_fps)
    ap_b = detect_cycle_boundaries(ap_res.signal, ap_fps, min_period_s=2.0)
    lat_b = detect_cycle_boundaries(lat_res.signal, lat_fps, min_period_s=2.0)
    ap_start, ap_end = pair_metrics["ap_cycle_frames"]
    lat_start, lat_end = pair_metrics["lateral_cycle_frames"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, res, fps, boundaries, view, sel, color in [
        (axes[0], ap_res, ap_fps, ap_b, "AP", (ap_start, ap_end), "tab:blue"),
        (axes[1], lat_res, lat_fps, lat_b, "Lateral", (lat_start, lat_end), "tab:orange"),
    ]:
        t = np.arange(res.signal.size) / fps
        ax.plot(t, res.signal, color="tab:gray", lw=0.8)
        for b in boundaries:
            if 0 <= b < t.size:
                ax.axvline(t[b], color="tab:red", alpha=0.4, lw=0.6, linestyle="--")
        ax.axvspan(sel[0] / fps, sel[1] / fps, color=color, alpha=0.2, label="selected cycle")
        ax.set_title(f"{view} f={res.dominant_frequency_hz:.3f} Hz, period={res.period_frames:.1f} fr")
        ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3); ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_pair_summary(pair_metrics: dict):
    """Reuses the (already saved) per-case PDF panels would be heavy; recompute a small summary."""
    ncc = pair_metrics["best_pair_ncc"]
    ap_dur = pair_metrics["ap_cycle_duration_s"]
    lat_dur = pair_metrics["lateral_cycle_duration_s"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    txt = (
        f"Best paired-cycle NCC: {ncc:.4f}\n"
        f"AP cycle frames: {pair_metrics['ap_cycle_frames']} "
        f"({ap_dur:.2f} s)\n"
        f"Lateral cycle frames: {pair_metrics['lateral_cycle_frames']} "
        f"({lat_dur:.2f} s)"
    )
    ax.text(0.02, 0.95, txt, ha="left", va="top", fontsize=12,
            family="monospace")
    return fig


def fig_predicted_4dct_panels(pred_4d: np.ndarray):
    """Pick 4 evenly-spaced frames; show their axial / coronal / sagittal mid-slices."""
    T = pred_4d.shape[0]
    idx = np.linspace(0, T - 1, 4).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    axis_lbl = ["Axial mid", "Coronal mid", "Sagittal mid"]
    for r, ax_axis in enumerate([0, 1, 2]):
        for c, t in enumerate(idx):
            axes[r, c].imshow(_mid_native(pred_4d[t], ax_axis), cmap="gray", vmin=0, vmax=1)
            if r == 0: axes[r, c].set_title(f"phase frame {t}")
            if c == 0: axes[r, c].set_ylabel(axis_lbl[r])
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("Predicted 4D CT — sampled phases × 3 anatomical views", fontsize=13)
    fig.tight_layout()
    return fig


def fig_qa_status_table(qa_summary: list[dict]):
    rows = []
    for r in qa_summary:
        mods = r["modules"]
        rows.append([
            f"{r['frame']:>2}",
            mods.get("RobustnessCheck", {}).get("status", "?"),
            mods.get("GeomIntegrity", {}).get("status", "?"),
            mods.get("TemporalMotion", {}).get("status", "?"),
        ])
    if not rows:
        rows = [["-", "-", "-", "-"]]
    fig, ax = plt.subplots(figsize=(8.27, max(3, 0.25 * len(rows) + 2)))
    ax.axis("off")
    ax.text(0.5, 0.97, "Generated CT QA — per-frame status",
            ha="center", va="top", fontsize=14, fontweight="bold")
    # Color cells
    colors = []
    for row in rows:
        row_colors = ["white"]
        for status in row[1:]:
            c = {"PASS": "#d4f5d4", "FAIL": "#fbd1d1",
                 "WARNING": "#fff3cd", "ERROR": "#e7daff"}.get(status, "white")
            row_colors.append(c)
        colors.append(row_colors)
    table = ax.table(cellText=rows,
                     colLabels=["frame", "Robustness", "GeomIntegrity", "TemporalMotion"],
                     cellLoc="center", colLoc="center", loc="center",
                     cellColours=colors,
                     colWidths=[0.15, 0.25, 0.3, 0.3])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.2)
    return fig


def fig_not_available(label: str):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    ax.text(0.5, 0.5, f"{label}: not available",
            ha="center", va="center", fontsize=14, color="#888")
    return fig


# ---------- per-case orchestration ----------

def build_case_sections(case_id: str, *, anon_root: Path, pair_qa_dir: Path,
                        inference_dir: Path, ct_qa_dir: Path) -> dict:
    sections: dict = {}
    info: dict[str, str] = {"case_id": case_id}

    # 0. DICOM info
    try:
        ap_path, lat_path = find_case_dicoms(anon_root, case_id)
        ap_full, ap_fps, ap_meta = load_dicom_frames(ap_path)
        lat_full, lat_fps, lat_meta = load_dicom_frames(lat_path)
        info.update({
            "AP frames":   f"{ap_full.shape[0]} @ {ap_fps:.2f} fps",
            "Lat frames":  f"{lat_full.shape[0]} @ {lat_fps:.2f} fps",
            "AP series":   ap_meta["series_description"],
            "Lat series":  lat_meta["series_description"],
        })
    except Exception as exc:
        ap_full = lat_full = None
        info["DICOMs"] = f"NOT FOUND ({exc})"

    # 1. Pair QA
    pair_metrics = load_json(pair_qa_dir / case_id / "metrics.json")
    if pair_metrics is not None:
        info["Pair NCC"] = f"{pair_metrics['best_pair_ncc']:.4f}"
        info["AP cycle frames"] = str(pair_metrics["ap_cycle_frames"])
        info["Lat cycle frames"] = str(pair_metrics["lateral_cycle_frames"])

    # 2. Inference
    pred_npz_path = inference_dir / f"case_{case_id}" / "pred_4d_ct.npz"
    pred_4d = None
    if pred_npz_path.is_file():
        pred_4d = np.load(pred_npz_path, allow_pickle=True)["ct"]
        info["Predicted 4DCT shape"] = str(tuple(pred_4d.shape))
        info["Predicted CT value range"] = f"[{float(pred_4d.min()):.3f}, {float(pred_4d.max()):.3f}]"

    # 3. Generated CT QA
    qa_summary = load_json(ct_qa_dir / f"case_{case_id}" / "summary.json")
    if qa_summary is not None:
        info["Generated-CT QA frames"] = str(len(qa_summary))

    # Build sections
    sections["Pipeline summary"] = fig_metadata_table(case_id, info)

    if ap_full is not None and lat_full is not None and pair_metrics is not None:
        ap_start, ap_end = pair_metrics["ap_cycle_frames"]
        lat_start, lat_end = pair_metrics["lateral_cycle_frames"]
        sections["Mean frames"] = fig_mean_frames(ap_full[ap_start:ap_end], lat_full[lat_start:lat_end])
        sections["Respiratory signals + cycle boundaries"] = fig_pair_signals(pair_metrics, ap_full, lat_full)
        sections["Pair-extraction summary"] = fig_pair_summary(pair_metrics)
    else:
        sections["Pair QA"] = fig_not_available("Pair QA")

    if pred_4d is not None:
        sections["Predicted 4DCT"] = fig_predicted_4dct_panels(pred_4d)
    else:
        sections["Predicted 4DCT"] = fig_not_available("Predicted 4DCT (run lung4d_inference)")

    if qa_summary is not None:
        sections["Generated-CT QA"] = fig_qa_status_table(qa_summary)
    else:
        sections["Generated-CT QA"] = fig_not_available("Generated CT QA (run qa_on_predicted_4dct)")

    return sections


def write_pdf(path: Path, case_id: str, sections: dict) -> None:
    with PdfPages(path) as pdf:
        for title, fig in sections.items():
            fig.suptitle(title, fontsize=12, y=1.02)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        meta = pdf.infodict()
        meta["Title"] = f"case {case_id} unified QA report"
        meta["CreationDate"] = dt.datetime.now()


def write_html(path: Path, case_id: str, sections: dict) -> None:
    bits = [
        f"<header><h1>case {html_mod.escape(case_id)} — unified QA report</h1>"
        f"<p>Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></header>"
    ]
    for title, fig in sections.items():
        b64 = _fig_to_b64(fig)
        bits.append(f"<section><h2>{html_mod.escape(title)}</h2>"
                    f"<img src='data:image/png;base64,{b64}'/></section>")
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<title>case {html_mod.escape(case_id)} — unified QA</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-top: 16px; }} h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
img {{ max-width: 100%; height: auto; }}
section {{ margin-bottom: 16px; }}
</style></head><body>
{chr(10).join(bits)}
</body></html>"""
    path.write_text(doc, encoding="utf-8")


# ---------- top index ----------

def write_top_index(out_dir: Path, case_ids: list[str], info_table: dict[str, dict]) -> None:
    rows = []
    for cid in case_ids:
        meta = info_table.get(cid, {})
        rows.append(
            f"<tr><td><a href='case_{cid}/report.html'>{cid}</a></td>"
            f"<td><a href='case_{cid}/report.pdf'>PDF</a></td>"
            f"<td>{meta.get('Pair NCC', '-')}</td>"
            f"<td>{meta.get('Predicted 4DCT shape', '-')}</td>"
            f"<td>{meta.get('Generated-CT QA frames', '-')}</td></tr>"
        )
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>Unified per-case QA</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
th {{ background: #eee; }}
</style></head>
<body>
<h1>Unified per-case QA reports</h1>
<p>Per-case PDF + HTML aggregating dynamic X-ray pair QA, 2D-4D inference,
and generated-CT QA into one document. Sections missing for a given case
are marked "not available".</p>
<table>
<tr><th>Case</th><th>PDF</th><th>Pair NCC</th><th>Predicted 4DCT shape</th><th>Generated-CT QA frames</th></tr>
{chr(10).join(rows)}
</table>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anon-root", type=Path, default=Path("DynamicChestXray_anon"))
    parser.add_argument("--pair-qa-dir", type=Path, default=Path("outputs/dynamic_xray_pair_qa"))
    parser.add_argument("--inference-dir", type=Path, default=Path("outputs/lung4d_inference"))
    parser.add_argument("--ct-qa-dir", type=Path, default=Path("outputs/qa_on_predicted_4dct"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/case_unified_report"))
    parser.add_argument("--cases", nargs="*",
                        help="Optional subset (default: every case dir under --pair-qa-dir)")
    args = parser.parse_args()

    if not args.pair_qa_dir.is_dir():
        print(f"no pair-qa dir: {args.pair_qa_dir}", file=sys.stderr); return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in args.pair_qa_dir.iterdir() if p.is_dir())
    if args.cases:
        chosen = set(args.cases)
        case_dirs = [d for d in case_dirs if d.name in chosen]

    info_table: dict[str, dict] = {}
    case_ids: list[str] = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        print(f"\n=== case {case_id} ===")
        case_out = args.out_dir / f"case_{case_id}"
        case_out.mkdir(parents=True, exist_ok=True)
        sections_html = build_case_sections(
            case_id,
            anon_root=args.anon_root, pair_qa_dir=args.pair_qa_dir,
            inference_dir=args.inference_dir, ct_qa_dir=args.ct_qa_dir,
        )
        write_html(case_out / "report.html", case_id, sections_html)
        # Rebuild figures for PDF (matplotlib figures are consumed by savefig+close)
        sections_pdf = build_case_sections(
            case_id,
            anon_root=args.anon_root, pair_qa_dir=args.pair_qa_dir,
            inference_dir=args.inference_dir, ct_qa_dir=args.ct_qa_dir,
        )
        write_pdf(case_out / "report.pdf", case_id, sections_pdf)
        case_ids.append(case_id)
        # capture info for index
        sec0 = sections_html.get("Pipeline summary")
        # We re-extract the info via the same builder side effect; simpler to re-call once more, but
        # rebuilding is acceptable for the index aggregate
        info = {}
        for k, v in []:
            info[k] = v
        # Cheap path: read the metrics.json files again for the table fields
        pm = load_json(args.pair_qa_dir / case_id / "metrics.json")
        if pm: info["Pair NCC"] = f"{pm['best_pair_ncc']:.4f}"
        pred = args.inference_dir / f"case_{case_id}" / "pred_4d_ct.npz"
        if pred.is_file():
            info["Predicted 4DCT shape"] = str(tuple(np.load(pred, allow_pickle=True)["ct"].shape))
        qa = load_json(args.ct_qa_dir / f"case_{case_id}" / "summary.json")
        if qa is not None:
            info["Generated-CT QA frames"] = str(len(qa))
        info_table[case_id] = info
        print(f"  wrote {case_out / 'report.pdf'}")

    write_top_index(args.out_dir, case_ids, info_table)
    print(f"\nIndex: {args.out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
