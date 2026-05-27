"""Tile 1-breath AP clips for every case into a single looping GIF.

Reads ``outputs/dynamic_xray_pair_qa/case<NN>/ap_cycle_frames.npy`` for each
case, time-resamples each clip to a common frame count, downsizes each frame
to a tile, lays the tiles out in a grid, and writes one animated GIF that
loops through the breath. Designed for quick visual triage — at a glance you
can see which cases have visible diaphragm excursion.

Usage::

    python scripts/diaphragm_motion_gif.py \\
        --pair-qa-dir outputs/dynamic_xray_pair_qa \\
        --out outputs/diaphragm_gif/all_cases_ap.gif

Patient-derived → output dir stays in .gitignore.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom


def load_case_clip(case_dir: Path) -> np.ndarray | None:
    p = case_dir / "ap_cycle_frames.npy"
    if not p.is_file():
        return None
    return np.load(p)


def time_resample(clip: np.ndarray, target_t: int) -> np.ndarray:
    """Linear time resample (T, H, W) -> (target_t, H, W)."""
    t_in = clip.shape[0]
    if t_in == target_t:
        return clip
    idx = np.linspace(0, t_in - 1, target_t).astype(np.float64)
    floor = np.floor(idx).astype(int)
    ceil = np.clip(floor + 1, 0, t_in - 1)
    frac = (idx - floor).reshape(-1, 1, 1)
    return (1.0 - frac) * clip[floor] + frac * clip[ceil]


def normalize_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile-normalize to uint8 grayscale."""
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def spatial_resize(frame_2d: np.ndarray, target: int) -> np.ndarray:
    h, w = frame_2d.shape
    z = (target / h, target / w)
    return zoom(frame_2d, z, order=1, mode="nearest")[:target, :target]


def annotate(frame_uint8: np.ndarray, text: str) -> np.ndarray:
    """Burn a tiny label into the top-left of the tile."""
    img = Image.fromarray(frame_uint8, mode="L").convert("RGB")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    # Black box + white text for legibility on dark X-rays.
    draw.rectangle((2, 2, 80, 22), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def tile_grid(tile_frames: list[np.ndarray], rows: int, cols: int, *,
              tile_size: int, gap: int = 4) -> np.ndarray:
    """Compose ``rows*cols`` tiles into a single RGB image with gaps."""
    grid_h = rows * tile_size + (rows - 1) * gap
    grid_w = cols * tile_size + (cols - 1) * gap
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for k, tile in enumerate(tile_frames):
        r, c = divmod(k, cols)
        y0 = r * (tile_size + gap)
        x0 = c * (tile_size + gap)
        canvas[y0:y0 + tile_size, x0:x0 + tile_size] = tile
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-qa-dir", type=Path, required=True,
                        help="Root containing case<NN>/ap_cycle_frames.npy")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output GIF path")
    parser.add_argument("--frames", type=int, default=32,
                        help="Common frame count after time resample (default 32)")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Side length per tile in pixels (default 256)")
    parser.add_argument("--cols", type=int, default=4,
                        help="Tile grid columns (default 4 -> 4x4 holds 14)")
    parser.add_argument("--frame-ms", type=int, default=100,
                        help="Per-frame display duration in milliseconds (default 100)")
    parser.add_argument("--cases", nargs="*",
                        help="Optional subset of case dir names")
    args = parser.parse_args()

    if not args.pair_qa_dir.is_dir():
        print(f"not a directory: {args.pair_qa_dir}", file=sys.stderr); return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in args.pair_qa_dir.iterdir() if p.is_dir() and (p / "ap_cycle_frames.npy").is_file())
    if args.cases:
        chosen = set(args.cases)
        case_dirs = [d for d in case_dirs if d.name in chosen]
    if not case_dirs:
        print(f"no ap_cycle_frames.npy under {args.pair_qa_dir}", file=sys.stderr); return 2

    n = len(case_dirs)
    cols = args.cols
    rows = (n + cols - 1) // cols
    print(f"Tiling {n} cases on a {rows}x{cols} grid, "
          f"frames={args.frames}, tile={args.tile_size}, frame_ms={args.frame_ms}")

    # Pre-process each case: load -> normalize -> resize -> time resample -> annotate
    per_case_frames: list[np.ndarray] = []  # each: (target_t, tile, tile, 3) uint8
    for case_dir in case_dirs:
        raw = load_case_clip(case_dir)
        if raw is None:
            print(f"  skip {case_dir.name}: no ap_cycle_frames.npy")
            continue
        clip = time_resample(raw, args.frames)  # (T, H, W) float
        out_frames = []
        for t in range(clip.shape[0]):
            small = spatial_resize(clip[t], args.tile_size)
            u8 = normalize_uint8(small)
            out_frames.append(annotate(u8, case_dir.name))
        per_case_frames.append(np.stack(out_frames))
        print(f"  {case_dir.name}: {raw.shape} -> ({args.frames}, {args.tile_size}, {args.tile_size}, 3)")

    # Pad with blank tiles so the grid is rectangular
    blank = np.zeros((args.frames, args.tile_size, args.tile_size, 3), dtype=np.uint8)
    while len(per_case_frames) < rows * cols:
        per_case_frames.append(blank)

    # Build a list of per-time grid frames
    print("Composing GIF frames ...")
    pil_frames: list[Image.Image] = []
    for t in range(args.frames):
        tiles_t = [case_seq[t] for case_seq in per_case_frames]
        grid = tile_grid(tiles_t, rows, cols, tile_size=args.tile_size)
        pil_frames.append(Image.fromarray(grid))

    pil_frames[0].save(
        str(args.out),
        save_all=True,
        append_images=pil_frames[1:],
        duration=args.frame_ms,
        loop=0,
        optimize=True,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nWrote {args.out}  ({size_mb:.1f} MB, {args.frames} frames @ {args.frame_ms}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
