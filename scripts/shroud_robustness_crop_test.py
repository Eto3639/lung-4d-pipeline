"""Test Amsterdam Shroud robustness when the diaphragm leaves the FOV.

Take an AP recording, crop the bottom by increasing amounts, and watch how
the respiratory signal / cycle detection degrades. The baseline (no crop)
has the diaphragm fully visible throughout; aggressive bottom-crops force
the diaphragm out of frame at end-inspiration so Shroud has to track a
ridge that may temporarily disappear.

Outputs (one case):
  - shroud_signals.png  — overlay of signals at every crop level
  - per-crop sample frames + shroud + signal panel
  - summary.json with f, period, #boundaries, NCC-vs-baseline
  - GIF showing all crop levels playing simultaneously

Usage::

    python scripts/shroud_robustness_crop_test.py \\
        --anon-root DynamicChestXray_anon \\
        --case 02 \\
        --out outputs/shroud_robustness/case_02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont

from dvf_qa.amsterdam_shroud import respiratory_signal, respiratory_signal_adaptive
from dvf_qa.cycle_pairing import detect_cycle_boundaries


DEFAULT_CROPS = (0.0, 0.15, 0.25, 0.35, 0.45)


def load_ap_frames(anon_root: Path, case_id: str) -> tuple[np.ndarray, float]:
    case_dir = anon_root / case_id
    files = sorted(p for p in case_dir.iterdir() if p.suffix == ".dcm")
    ap_path = files[0]  # convention: rec1 = AP
    ds = pydicom.dcmread(str(ap_path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    slope = float(ds.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(ds.get("RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    if str(ds.get("PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
        arr = arr.max() - arr
    fps = 1000.0 / float(ds.get("FrameTime", 66.0) or 66.0)
    return arr, fps


def crop_bottom(frames: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0:
        return frames
    h = frames.shape[1]
    keep_h = int(round(h * (1.0 - fraction)))
    keep_h = max(8, keep_h)
    return frames[:, :keep_h, :]


def boundary_match_score(baseline: np.ndarray, candidate: np.ndarray, *, fps: float,
                          tol_s: float = 0.5) -> float:
    """Fraction of baseline cycle boundaries that find a candidate within ``tol_s``."""
    if baseline.size == 0:
        return float("nan")
    tol = max(1, int(round(tol_s * fps)))
    hits = 0
    for b in baseline:
        if np.any(np.abs(candidate - b) <= tol):
            hits += 1
    return hits / baseline.size


def run_one_crop(frames_full: np.ndarray, fps: float, fraction: float,
                 *, strategy: str = "fixed") -> dict:
    cropped = crop_bottom(frames_full, fraction)
    band: tuple[float, float] | None = None
    if strategy == "adaptive":
        res, band = respiratory_signal_adaptive(cropped, fps=fps, return_band=True)
    else:
        res = respiratory_signal(cropped, fps=fps)
    boundaries = detect_cycle_boundaries(res.signal, fps, min_period_s=2.0)
    return {
        "crop_fraction": fraction,
        "strategy": strategy,
        "search_band": band,
        "input_height": int(cropped.shape[1]),
        "shroud": res.shroud,
        "diaphragm_row": res.diaphragm_row,
        "signal": np.asarray(res.signal),
        "dominant_frequency_hz": float(res.dominant_frequency_hz),
        "period_frames": float(res.period_frames),
        "boundaries": np.asarray(boundaries, dtype=int),
        "cropped_frames": cropped,
    }


# ---------- figures ----------

def fig_signals_overlay(results: list[dict], fps: float):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(results)))
    for r, c in zip(results, colors):
        sig = r["signal"]
        sd = max(1e-9, np.std(sig))
        ax.plot(np.arange(sig.size) / fps, sig / sd,
                color=c, lw=1.0,
                label=f"crop={r['crop_fraction']*100:.0f}%  "
                       f"f={r['dominant_frequency_hz']:.3f}Hz  "
                       f"#bnd={len(r['boundaries'])}")
        for b in r["boundaries"]:
            if 0 <= b < sig.size:
                ax.axvline(b / fps, color=c, alpha=0.25, lw=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("z-scored signal")
    ax.set_title("Shroud signals across crop levels (vertical bars = detected end-expiration)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_summary_table(rows: list[list[str]]):
    fig, ax = plt.subplots(figsize=(8.27, max(2, 0.3 * len(rows) + 2)))
    ax.axis("off")
    ax.text(0.5, 0.95, "Crop robustness summary", ha="center", va="top",
            fontsize=15, fontweight="bold")
    headers = ["crop %", "height", "f (Hz)", "period (fr)", "#boundaries", "boundary recall vs baseline"]
    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc="center", colLoc="center", loc="center",
                     colWidths=[0.1, 0.12, 0.13, 0.14, 0.16, 0.3])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.6)
    return fig


def fig_summary_table_compare(rows: list[list[str]]):
    """Side-by-side fixed vs adaptive comparison table."""
    fig, ax = plt.subplots(figsize=(13, max(2.5, 0.32 * len(rows) + 2)))
    ax.axis("off")
    ax.text(0.5, 0.95, "Crop robustness  —  fixed vs adaptive search_band",
            ha="center", va="top", fontsize=15, fontweight="bold")
    headers = [
        "crop %", "height",
        "FIXED f (Hz)", "FIXED #bnd", "FIXED recall",
        "ADP f (Hz)", "ADP #bnd", "ADP recall", "ADP band",
    ]
    table = ax.table(cellText=rows, colLabels=headers,
                     cellLoc="center", colLoc="center", loc="center",
                     colWidths=[0.07, 0.08, 0.1, 0.09, 0.1, 0.1, 0.09, 0.1, 0.12])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.7)
    # color recall cells
    for ri, row in enumerate(rows, start=1):  # +1 to skip header row
        for ci, val in [(4, row[4]), (7, row[7])]:
            try:
                v = float(val)
                if v >= 0.8: c = "#d4f5d4"
                elif v >= 0.5: c = "#fff3cd"
                else: c = "#fbd1d1"
                table[(ri, ci)].set_facecolor(c)
            except ValueError:
                pass
    return fig


def fig_per_crop_panel(r: dict, fps: float):
    cropped = r["cropped_frames"]
    sig = r["signal"]
    # sample 3 frames: argmin / mid / argmax of the signal (rough end-exp / mid / end-insp)
    idx_min = int(np.argmin(sig))
    idx_max = int(np.argmax(sig))
    idx_mid = sig.size // 2
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])
    # Top row: 3 sample frames
    for c, (idx, lbl) in enumerate(zip([idx_min, idx_mid, idx_max],
                                        ["end-exp (argmin)", "mid", "end-insp (argmax)"])):
        ax = fig.add_subplot(grid[0, c])
        ax.imshow(cropped[idx], cmap="gray", aspect="equal")
        ax.set_title(f"frame {idx} — {lbl}")
        ax.set_xticks([]); ax.set_yticks([])
    # Bottom: shroud + signal
    ax_shroud = fig.add_subplot(grid[1, 0:2])
    ax_shroud.imshow(r["shroud"], aspect="auto", cmap="gray", origin="upper")
    ax_shroud.plot(np.arange(r["diaphragm_row"].size), r["diaphragm_row"], color="tab:red", lw=1.0)
    ax_shroud.set_title("Shroud + tracked diaphragm")
    ax_shroud.set_xlabel("Frame"); ax_shroud.set_ylabel("CC position (px)")
    ax_sig = fig.add_subplot(grid[1, 2])
    t = np.arange(sig.size) / fps
    ax_sig.plot(t, sig, color="tab:blue", lw=0.9)
    for b in r["boundaries"]:
        if 0 <= b < sig.size:
            ax_sig.axvline(b / fps, color="tab:red", alpha=0.5, lw=0.6, linestyle="--")
    ax_sig.set_title(f"signal  f={r['dominant_frequency_hz']:.3f} Hz  "
                      f"period={r['period_frames']:.1f} fr  #b={len(r['boundaries'])}")
    ax_sig.set_xlabel("Time (s)"); ax_sig.grid(True, alpha=0.3)
    fig.suptitle(f"crop={r['crop_fraction']*100:.0f}%  (input height {r['input_height']})",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    return fig


def annotate_tile(frame_uint8: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame_uint8, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((2, 2, 130, 22), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def make_robustness_gif(results: list[dict], out_path: Path, *, fps: float,
                        target_h: int = 320, frame_ms: int = 100, n_frames: int = 32):
    """Side-by-side GIF: each crop level animates simultaneously through one breath."""
    tiles_per_crop: list[list[np.ndarray]] = []
    for r in results:
        clip = r["cropped_frames"]
        # time-resample
        idx = np.linspace(0, clip.shape[0] - 1, n_frames).astype(np.float64)
        floor = np.floor(idx).astype(int)
        ceil = np.clip(floor + 1, 0, clip.shape[0] - 1)
        frac = (idx - floor).reshape(-1, 1, 1)
        resampled = (1.0 - frac) * clip[floor] + frac * clip[ceil]
        # spatial resize keeping aspect (height fixed)
        h, w = resampled.shape[1], resampled.shape[2]
        scale = target_h / h
        new_w = max(1, int(round(w * scale)))
        out_seq = []
        for t in range(n_frames):
            from scipy.ndimage import zoom
            small = zoom(resampled[t], (scale, scale), order=1)
            small = small[:target_h, :new_w]
            lo, hi = np.percentile(small, [1, 99])
            u8 = np.clip((small - lo) / max(1e-9, hi - lo) * 255, 0, 255).astype(np.uint8)
            label = f"crop={r['crop_fraction']*100:.0f}%"
            out_seq.append(annotate_tile(u8, label))
        tiles_per_crop.append(out_seq)
    # Compose: horizontal layout
    pil_frames: list[Image.Image] = []
    gap = 6
    total_w = sum(seq[0].shape[1] for seq in tiles_per_crop) + gap * (len(tiles_per_crop) - 1)
    for t in range(n_frames):
        canvas = np.zeros((target_h, total_w, 3), dtype=np.uint8)
        x = 0
        for seq in tiles_per_crop:
            tile = seq[t]
            canvas[:, x:x + tile.shape[1]] = tile
            x += tile.shape[1] + gap
        pil_frames.append(Image.fromarray(canvas))
    pil_frames[0].save(str(out_path), save_all=True, append_images=pil_frames[1:],
                       duration=frame_ms, loop=0, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anon-root", type=Path, default=Path("DynamicChestXray_anon"))
    parser.add_argument("--case", required=True, help="case id (e.g. 02)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crops", type=float, nargs="*", default=list(DEFAULT_CROPS),
                        help="Crop fractions (default: 0 0.15 0.25 0.35 0.45)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Loading AP recording for case {args.case}")
    frames, fps = load_ap_frames(args.anon_root, args.case)
    print(f"  shape={frames.shape}, fps={fps:.2f}")

    print("\nRunning Amsterdam Shroud at each crop level (fixed search_band) ...")
    results: list[dict] = []
    for c in args.crops:
        r = run_one_crop(frames, fps, c, strategy="fixed")
        print(f"  fixed   crop={c*100:>4.1f}%  height={r['input_height']:>4}  "
              f"f={r['dominant_frequency_hz']:.3f}Hz  #bnd={len(r['boundaries'])}")
        results.append(r)

    print("\nRunning Amsterdam Shroud at each crop level (adaptive search_band) ...")
    results_adaptive: list[dict] = []
    for c in args.crops:
        r = run_one_crop(frames, fps, c, strategy="adaptive")
        band = r["search_band"]
        band_str = f"({band[0]:.2f}, {band[1]:.2f})" if band else "n/a"
        print(f"  adapt   crop={c*100:>4.1f}%  height={r['input_height']:>4}  "
              f"band={band_str}  f={r['dominant_frequency_hz']:.3f}Hz  #bnd={len(r['boundaries'])}")
        results_adaptive.append(r)

    baseline = results[0]
    rows = []
    summary_json: list[dict] = []
    for r_fixed, r_adp in zip(results, results_adaptive):
        recall_fixed = boundary_match_score(baseline["boundaries"], r_fixed["boundaries"], fps=fps)
        recall_adp = boundary_match_score(baseline["boundaries"], r_adp["boundaries"], fps=fps)
        band = r_adp["search_band"]
        band_str = f"({band[0]:.2f},{band[1]:.2f})" if band else "n/a"
        rows.append([
            f"{r_fixed['crop_fraction']*100:.0f}",
            f"{r_fixed['input_height']}",
            f"{r_fixed['dominant_frequency_hz']:.3f}",
            f"{len(r_fixed['boundaries'])}",
            f"{recall_fixed:.2f}" if recall_fixed == recall_fixed else "n/a",
            f"{r_adp['dominant_frequency_hz']:.3f}",
            f"{len(r_adp['boundaries'])}",
            f"{recall_adp:.2f}" if recall_adp == recall_adp else "n/a",
            band_str,
        ])
        summary_json.append({
            "crop_fraction": r_fixed["crop_fraction"],
            "input_height": r_fixed["input_height"],
            "fixed": {
                "dominant_frequency_hz": r_fixed["dominant_frequency_hz"],
                "n_boundaries": len(r_fixed["boundaries"]),
                "boundaries": r_fixed["boundaries"].tolist(),
                "boundary_recall_vs_baseline": float(recall_fixed) if recall_fixed == recall_fixed else None,
            },
            "adaptive": {
                "search_band": list(band) if band else None,
                "dominant_frequency_hz": r_adp["dominant_frequency_hz"],
                "n_boundaries": len(r_adp["boundaries"]),
                "boundaries": r_adp["boundaries"].tolist(),
                "boundary_recall_vs_baseline": float(recall_adp) if recall_adp == recall_adp else None,
            },
        })

    fig_summary_table_compare(rows).savefig(args.out / "summary_table.png",
                                            dpi=130, bbox_inches="tight"); plt.close("all")
    fig_signals_overlay(results, fps).savefig(args.out / "signals_fixed.png",
                                              dpi=130, bbox_inches="tight"); plt.close("all")
    fig_signals_overlay(results_adaptive, fps).savefig(args.out / "signals_adaptive.png",
                                                       dpi=130, bbox_inches="tight"); plt.close("all")
    for r in results:
        fname = f"fixed_crop_{int(r['crop_fraction']*100):02d}.png"
        fig_per_crop_panel(r, fps).savefig(args.out / fname, dpi=130, bbox_inches="tight"); plt.close("all")
    for r in results_adaptive:
        fname = f"adaptive_crop_{int(r['crop_fraction']*100):02d}.png"
        fig_per_crop_panel(r, fps).savefig(args.out / fname, dpi=130, bbox_inches="tight"); plt.close("all")
    make_robustness_gif(results, args.out / "side_by_side_fixed.gif", fps=fps)
    make_robustness_gif(results_adaptive, args.out / "side_by_side_adaptive.gif", fps=fps)

    (args.out / "summary.json").write_text(
        json.dumps({
            "case_id": args.case, "fps": fps,
            "input_full_shape": list(frames.shape),
            "results": summary_json,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote:\n  {args.out}/summary_table.png\n  {args.out}/signals_overlay.png\n"
          f"  {args.out}/crop_*.png\n  {args.out}/side_by_side.gif\n  {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
