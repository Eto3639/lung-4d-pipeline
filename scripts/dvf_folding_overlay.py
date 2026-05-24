"""Render anatomical overlays of where folding (J<=0) occurs in predicted DVFs.

For each phase, takes the moving CT (phase 50 = ``DVF_predict/3D/``), the
predicted DVF, computes the Jacobian determinant on the interior (edge_crop
voxels excluded to suppress boundary artifacts), and overlays the folding
mask in red on top of the CT mid-slices (axial / coronal / sagittal).

Also produces a per-axis "where does folding concentrate" projection (max
along the third axis of the folding mask) so it is easy to spot whether
the folding is in the lung, mediastinum, or outside the body silhouette.

Reports go to ``outputs/dvf_folding_overlay/`` and stay patient-data-safe:
phantom-derived (DVF_predict already gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from dvf_qa.jacobian import jacobian_determinant


PHASES = ["00", "10", "20", "30", "40", "60", "70", "80", "90"]


def read_dicom_dir(d: Path) -> np.ndarray:
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(d))
    if not ids:
        raise FileNotFoundError(f"No DICOM series under {d}")
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(d), ids[0]))
    return sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)


def load_dvf(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = tuple(float(s) for s in img.GetSpacing()[:3])
    return arr, spacing


def interior_mask(shape: tuple[int, int, int], crop: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    z, y, x = shape
    m[crop:z - crop, crop:y - crop, crop:x - crop] = True
    return m


def _mid(volume: np.ndarray, axis: int) -> np.ndarray:
    sl = np.take(volume, volume.shape[axis] // 2, axis=axis)
    if axis in (1, 2):
        sl = np.flipud(sl)
    return sl


def _proj_max(volume: np.ndarray, axis: int) -> np.ndarray:
    sl = volume.max(axis=axis)
    if axis in (1, 2):
        sl = np.flipud(sl)
    return sl


def overlay_red_on_gray(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    g = (gray - gray.min()) / max(1e-9, gray.max() - gray.min())
    rgb = np.stack([g, g, g], axis=-1)
    if mask.any():
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.array([1.0, 0.1, 0.1])
    return np.clip(rgb, 0.0, 1.0)


def render_phase(phase: str, root: Path, moving: np.ndarray, edge_crop: int, out_dir: Path) -> dict:
    print(f"\n=== phase {phase} ===")
    dvf, spacing = load_dvf(root / "DVF" / f"prediction_{phase}.nii.gz")
    interior = interior_mask(dvf.shape[:3], crop=edge_crop)
    jac = jacobian_determinant(dvf, spacing)
    folding = (jac <= 0) & interior
    fold_frac = float(folding.sum()) / float(interior.sum())
    voxel_ml = float(np.prod(spacing) / 1000.0)
    fold_ml = float(folding.sum()) * voxel_ml

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axis_labels = ["Axial", "Coronal", "Sagittal"]
    for r, ax_axis in enumerate([0, 1, 2]):
        ct_slice = _mid(moving, ax_axis)
        fold_slice = _mid(folding.astype(bool), ax_axis)
        fold_proj = _proj_max(folding.astype(bool), ax_axis)

        axes[r, 0].imshow(ct_slice, cmap="gray")
        axes[r, 0].set_title(f"{axis_labels[r]} mid CT")
        axes[r, 1].imshow(overlay_red_on_gray(ct_slice, fold_slice))
        axes[r, 1].set_title(f"{axis_labels[r]} mid CT + folding (this slice)")
        axes[r, 2].imshow(overlay_red_on_gray(ct_slice, fold_proj))
        axes[r, 2].set_title(f"{axis_labels[r]} CT + max-proj folding (any depth)")
        axes[r, 3].imshow(fold_proj, cmap="Reds", vmin=0, vmax=1)
        axes[r, 3].set_title(f"{axis_labels[r]} folding max-projection")
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle(f"phase {phase} — folding (J≤0) anatomical overlay  "
                 f"|  fraction={fold_frac:.3e}, total={fold_ml:.2f} mL  "
                 f"(edge_crop={edge_crop})", fontsize=12, y=1.00)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase_{phase}_folding.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  fraction J<=0 = {fold_frac:.3e}, fold volume = {fold_ml:.2f} mL")
    print(f"  saved {out_path}")
    return {
        "phase": phase,
        "fold_fraction": fold_frac,
        "fold_volume_ml": fold_ml,
        "image": str(out_path.name),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="DVF_predict directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phases", nargs="*", default=PHASES)
    parser.add_argument("--edge-crop", type=int, default=3)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print("Loading moving CT (phase 50) ...")
    moving = read_dicom_dir(args.root / "3D")
    print(f"  shape={moving.shape}")

    results: list[dict] = []
    for phase in args.phases:
        try:
            results.append(render_phase(phase, args.root, moving=moving,
                                        edge_crop=args.edge_crop, out_dir=args.out))
        except Exception as exc:
            print(f"FAILED phase {phase}: {exc}", file=sys.stderr)

    # Build a contact-sheet index
    rows = "\n".join(
        f"<div class='card'><h2>phase {r['phase']}</h2>"
        f"<p>fold frac=<b>{r['fold_fraction']:.3e}</b>, "
        f"fold vol=<b>{r['fold_volume_ml']:.2f} mL</b></p>"
        f"<img src='{r['image']}'/></div>"
        for r in results
    )
    (args.out / "index.html").write_text(
        f"""<!DOCTYPE html><html lang='ja'><head><meta charset='utf-8'>
<title>DVF folding anatomical overlay</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
.card {{ margin-bottom: 32px; border-bottom: 1px solid #ddd; padding-bottom: 16px; }}
img {{ max-width: 100%; height: auto; }}
</style></head><body>
<h1>DVF folding anatomical overlay</h1>
<p>Red overlay = folding voxels (J ≤ 0). The third column is a max-projection
across the depth axis so any folding anywhere along that axis appears even
if it is not on the displayed mid-slice.</p>
{rows}
</body></html>""",
        encoding="utf-8",
    )
    (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nIndex: {args.out / 'index.html'}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
