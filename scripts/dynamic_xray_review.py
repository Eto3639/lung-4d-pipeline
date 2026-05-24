"""Build a one-page side-by-side review of sub-threshold pair-QA cases.

Walks ``outputs/dynamic_xray_pair_qa/summary.json``, selects cases with
NCC below ``--threshold`` (default 0.95), and renders a stacked image that
shows for each flagged case:

  - case id + NCC + per-view dominant frequency
  - AP and Lateral mean frame
  - both respiratory signals with cycle boundaries and selected windows
  - resampled selected-pair overlay

The output PNG is meant to be opened once and scrolled top-to-bottom for
a quick "do these auto-extracted pairs look right?" check. It does not
embed in HTML and is not deployed (patient-derived).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom

from dvf_qa.amsterdam_shroud import respiratory_signal
from dvf_qa.cycle_pairing import (
    detect_cycle_boundaries,
    pair_best_correlated_cycles,
)


def load_dicom_frames(path: Path) -> tuple[np.ndarray, float]:
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
    return arr, fps


def find_case_dicoms(anon_root: Path, case_id: str) -> tuple[Path, Path]:
    case_dir = anon_root / case_id
    files = sorted(p for p in case_dir.iterdir() if p.suffix == ".dcm")
    if len(files) < 2:
        raise FileNotFoundError(f"need 2 DICOMs under {case_dir}, found {len(files)}")
    return files[0], files[1]


def render_case_row(axes, case_id: str, ap_path: Path, lat_path: Path, target_ncc: float | None):
    """Render a single horizontal strip (1 row x 4 cols) for one case."""
    ap_frames, ap_fps = load_dicom_frames(ap_path)
    lat_frames, lat_fps = load_dicom_frames(lat_path)
    ap_res = respiratory_signal(ap_frames, fps=ap_fps)
    lat_res = respiratory_signal(lat_frames, fps=lat_fps)
    ap_b = detect_cycle_boundaries(ap_res.signal, ap_fps, min_period_s=2.0)
    lat_b = detect_cycle_boundaries(lat_res.signal, lat_fps, min_period_s=2.0)
    pair = pair_best_correlated_cycles(ap_res.signal, lat_res.signal,
                                       ap_fps=ap_fps, lateral_fps=lat_fps,
                                       min_period_s=2.0)

    ax_ap_img, ax_lat_img, ax_signals, ax_overlay = axes
    ax_ap_img.imshow(ap_frames.mean(0), cmap="gray")
    ax_ap_img.set_title(f"case {case_id} — AP mean", fontsize=10)
    ax_ap_img.set_xticks([]); ax_ap_img.set_yticks([])

    ax_lat_img.imshow(lat_frames.mean(0), cmap="gray")
    ax_lat_img.set_title("Lateral mean", fontsize=10)
    ax_lat_img.set_xticks([]); ax_lat_img.set_yticks([])

    t_ap = np.arange(ap_res.signal.size) / ap_fps
    t_lat = np.arange(lat_res.signal.size) / lat_fps
    ax_signals.plot(t_ap, ap_res.signal / max(1e-9, np.std(ap_res.signal)), color="tab:blue", lw=0.9, label="AP (z)")
    ax_signals.plot(t_lat, lat_res.signal / max(1e-9, np.std(lat_res.signal)), color="tab:orange", lw=0.9, label="Lateral (z)")
    for b in ap_b: ax_signals.axvline(t_ap[b], color="tab:blue", alpha=0.25, lw=0.5)
    for b in lat_b: ax_signals.axvline(t_lat[b], color="tab:orange", alpha=0.25, lw=0.5)
    ax_signals.axvspan(pair.ap_start / ap_fps, pair.ap_end / ap_fps, color="tab:blue", alpha=0.15)
    ax_signals.axvspan(pair.lateral_start / lat_fps, pair.lateral_end / lat_fps, color="tab:orange", alpha=0.15)
    ax_signals.set_title(f"signals  AP f={ap_res.dominant_frequency_hz:.3f}  Lat f={lat_res.dominant_frequency_hz:.3f}",
                         fontsize=10)
    ax_signals.set_xlabel("Time (s)", fontsize=8)
    ax_signals.legend(fontsize=7, loc="upper right")
    ax_signals.grid(True, alpha=0.3)

    x = np.linspace(0.0, 1.0, pair.resample_length)
    ax_overlay.plot(x, pair.ap_resampled, color="tab:blue", lw=1.0, label="AP")
    ax_overlay.plot(x, pair.lateral_resampled, color="tab:orange", lw=1.0, label="Lat")
    ax_overlay.set_title(f"paired cycle  NCC={pair.correlation:.3f}", fontsize=10)
    ax_overlay.set_xlabel("cycle fraction", fontsize=8)
    ax_overlay.legend(fontsize=7); ax_overlay.grid(True, alpha=0.3)
    return pair.correlation, ap_res.dominant_frequency_hz, lat_res.dominant_frequency_hz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anon-root", type=Path, required=True,
                        help="DynamicChestXray_anon/")
    parser.add_argument("--summary", type=Path, required=True,
                        help="outputs/dynamic_xray_pair_qa/summary.json")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output PNG (e.g. outputs/dynamic_xray_review/review.png)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Flag cases with NCC below this (default 0.95)")
    parser.add_argument("--cases", nargs="*",
                        help="Force-include these case ids regardless of NCC")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    flagged: list[dict] = []
    forced = set(args.cases or [])
    for r in summary:
        if r["case_id"] in forced or r.get("best_pair_ncc", 1.0) < args.threshold:
            flagged.append(r)
    flagged.sort(key=lambda r: r.get("best_pair_ncc", 1.0))
    if not flagged:
        print(f"No cases below NCC {args.threshold}. Nothing to render.")
        return 0
    print(f"Reviewing {len(flagged)} cases: {[r['case_id'] for r in flagged]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(flagged), 4, figsize=(18, 4 * len(flagged)))
    if len(flagged) == 1:
        axes = axes[None, :]
    for row, rec in zip(axes, flagged):
        ap_path, lat_path = find_case_dicoms(args.anon_root, rec["case_id"])
        ncc, _, _ = render_case_row(row, rec["case_id"], ap_path, lat_path, rec.get("best_pair_ncc"))
        print(f"  rendered case {rec['case_id']}: NCC={ncc:.3f}")

    fig.suptitle(f"Dynamic chest X-ray pair QA review (NCC < {args.threshold})", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
