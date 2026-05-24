"""Convert per-case AP/LAT one-breath cycles into model-ready tensors.

Reads ``outputs/dynamic_xray_pair_qa/case<NN>/{ap,lateral}_cycle_frames.npy``,
resamples each clip to a common frame count and spatial resolution, and
writes a single ``paired_dataset.npz`` plus per-case ``.npz`` files for
manual inspection.

Typical use::

    python scripts/prep_2d4d_inputs.py \\
        --root outputs/dynamic_xray_pair_qa \\
        --out outputs/lung4d_input \\
        --frames 32 --size 256

The output bundle stores AP and LAT as ``(N, T, H, W)`` float32 in
``[0, 1]`` plus a list of case ids. Load it from your 2D-4D model with::

    import numpy as np
    bundle = np.load("outputs/lung4d_input/paired_dataset.npz",
                     allow_pickle=True)
    ap, lat, ids = bundle["ap"], bundle["lateral"], bundle["case_ids"]
    # ap.shape == (14, T, H, W), lateral.shape == (14, T, H, W)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def resample_clip(clip: np.ndarray, *, frames: int, size: int) -> np.ndarray:
    """Resample ``(T, H, W)`` to ``(frames, size, size)``, linear interpolation."""
    if clip.ndim != 3:
        raise ValueError(f"expected (T, H, W); got {clip.shape}")
    t_in, h_in, w_in = clip.shape
    factors = (frames / t_in, size / h_in, size / w_in)
    out = zoom(clip.astype(np.float32), factors, order=1, mode="nearest")
    # zoom can occasionally produce off-by-one in the output dims; crop/pad to target.
    if out.shape != (frames, size, size):
        out = out[:frames, :size, :size]
        if out.shape != (frames, size, size):
            padded = np.zeros((frames, size, size), dtype=np.float32)
            padded[:out.shape[0], :out.shape[1], :out.shape[2]] = out
            out = padded
    return out


def normalize_minmax(clip: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(clip, 1))
    hi = float(np.percentile(clip, 99))
    if hi <= lo:
        return np.zeros_like(clip, dtype=np.float32)
    return np.clip((clip.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def process_case(case_dir: Path, *, frames: int, size: int) -> dict:
    ap_path = case_dir / "ap_cycle_frames.npy"
    lat_path = case_dir / "lateral_cycle_frames.npy"
    if not (ap_path.is_file() and lat_path.is_file()):
        raise FileNotFoundError(f"missing AP or LAT cycle in {case_dir}")
    ap_raw = np.load(ap_path)
    lat_raw = np.load(lat_path)
    ap = normalize_minmax(resample_clip(ap_raw, frames=frames, size=size))
    lat = normalize_minmax(resample_clip(lat_raw, frames=frames, size=size))
    return {
        "case_id": case_dir.name,
        "ap": ap,
        "lateral": lat,
        "src_ap_shape": list(ap_raw.shape),
        "src_lateral_shape": list(lat_raw.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="dynamic_xray_pair_qa outputs root (case<NN>/ children)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32,
                        help="Target number of frames per cycle (default 32)")
    parser.add_argument("--size", type=int, default=256,
                        help="Target spatial size HxW (default 256)")
    parser.add_argument("--cases", nargs="*",
                        help="Optional subset of case directory names")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(p for p in args.root.iterdir() if p.is_dir() and (p / "ap_cycle_frames.npy").is_file())
    if args.cases:
        chosen = set(args.cases)
        case_dirs = [d for d in case_dirs if d.name in chosen]
    if not case_dirs:
        print(f"No usable case dirs under {args.root}", file=sys.stderr); return 2

    ap_stack: list[np.ndarray] = []
    lat_stack: list[np.ndarray] = []
    case_ids: list[str] = []
    audit: list[dict] = []

    for case_dir in case_dirs:
        try:
            rec = process_case(case_dir, frames=args.frames, size=args.size)
        except Exception as exc:
            print(f"FAILED {case_dir.name}: {exc}", file=sys.stderr)
            continue
        # per-case npz for manual loading
        np.savez_compressed(
            args.out / f"{rec['case_id']}.npz",
            ap=rec["ap"], lateral=rec["lateral"], case_id=rec["case_id"],
        )
        ap_stack.append(rec["ap"])
        lat_stack.append(rec["lateral"])
        case_ids.append(rec["case_id"])
        audit.append({
            "case_id": rec["case_id"],
            "src_ap_shape": rec["src_ap_shape"],
            "src_lateral_shape": rec["src_lateral_shape"],
            "out_shape": [args.frames, args.size, args.size],
        })
        print(f"  {rec['case_id']}: AP {rec['src_ap_shape']} -> "
              f"({args.frames},{args.size},{args.size}), value range "
              f"[{rec['ap'].min():.3f}, {rec['ap'].max():.3f}]")

    bundle = {
        "ap": np.stack(ap_stack),                 # (N, T, H, W)
        "lateral": np.stack(lat_stack),           # (N, T, H, W)
        "case_ids": np.array(case_ids),
    }
    np.savez_compressed(args.out / "paired_dataset.npz", **bundle)

    spec = {
        "n_cases": len(case_ids),
        "frames_per_cycle": int(args.frames),
        "spatial_size": int(args.size),
        "dtype": "float32",
        "value_range": "[0, 1] per case (percentile 1-99 min-max)",
        "ap_shape": list(bundle["ap"].shape),
        "lateral_shape": list(bundle["lateral"].shape),
        "case_ids": case_ids,
        "audit": audit,
    }
    (args.out / "dataset_spec.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote bundle: {args.out / 'paired_dataset.npz'}  "
          f"AP={bundle['ap'].shape}  LAT={bundle['lateral'].shape}")
    print(f"Spec: {args.out / 'dataset_spec.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
