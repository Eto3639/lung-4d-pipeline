"""DVF QA on the DVF_predict/ directory.

For each predicted DVF (phase 50 -> phase PP) under ``DVF_predict/DVF/``,
this script computes the canonical DVF quality metrics and produces a
per-phase PDF + HTML report:

  * Jacobian determinant of T(x) = x + u(x), folding fraction, percentiles
  * DVF magnitude, gradient norm, bending energy
  * Warped CT (moving = phase 50 from ``3D/``) vs fixed CT
    (target phase from ``4D/CT_PP_*.dcm``) similarity (MSE / MAE / NCC)
  * Cross-check: our warping result vs ``Other/deformed_image_50toPP.nii.gz``

A top-level ``index.html`` + ``summary.json`` aggregate the runs and become
the entry point for the GitHub Pages deploy.

Usage::

    python scripts/dvf_predict_qa.py --root DVF_predict --out-dir outputs/dvf_predict_qa
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
import SimpleITK as sitk
from matplotlib.backends.backend_pdf import PdfPages

from dvf_qa.metrics import image_similarity, summarize_dvf_qa
from dvf_qa.warp import warp_volume_moving_to_fixed


PHASES = ["00", "10", "20", "30", "40", "60", "70", "80", "90"]


# ---------- I/O helpers ----------

def load_dvf(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected DVF (z,y,x,3); got {arr.shape}")
    spacing = tuple(float(s) for s in img.GetSpacing()[:3])
    return arr, spacing


def read_dicom_dir(d: Path) -> np.ndarray:
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(d))
    if not ids:
        raise FileNotFoundError(f"No DICOM series under {d}")
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(d), ids[0]))
    return sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)


def load_4d_phase(four_d_dir: Path, phase_tag: str) -> np.ndarray:
    files = sorted(four_d_dir.glob(f"CT_{phase_tag}_*.dcm"))
    if not files:
        raise FileNotFoundError(f"No CT_{phase_tag}_*.dcm under {four_d_dir}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in files])
    return sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)


def load_nifti(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


# ---------- figure helpers ----------

def _norm(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    a = np.nan_to_num(img.astype(np.float32))
    lo, hi = np.percentile(a, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _mid(volume: np.ndarray, axis: int) -> np.ndarray:
    """Return the mid-slice of ``volume`` along ``axis``.

    For coronal (axis=1) and sagittal (axis=2) slices, after :func:`numpy.take`
    the first remaining dimension is the cranio-caudal (Z) axis. SimpleITK
    typically returns DICOMs with row 0 at the inferior side, so without a flip
    the head appears at the bottom of the image. Apply :func:`numpy.flipud`
    on those views so the head is up.
    """
    sl = np.take(volume, volume.shape[axis] // 2, axis=axis)
    if axis in (1, 2):
        sl = np.flipud(sl)
    return sl


def make_interior_mask(shape: tuple[int, ...], crop: int) -> np.ndarray:
    """Return a boolean mask that excludes ``crop`` voxels from each side.

    Slice-edge voxels suffer from one-sided gradient approximations and DVF
    boundary discontinuities; excluding them gives QA metrics that reflect the
    interior of the volume rather than ringing at the data boundary.
    """
    mask = np.zeros(shape, dtype=bool)
    if crop <= 0:
        mask[...] = True
        return mask
    z, y, x = shape
    mask[crop:z - crop, crop:y - crop, crop:x - crop] = True
    return mask


def fig_ct_comparison(moving: np.ndarray, fixed: np.ndarray, warped: np.ndarray, phase: str):
    """3-axis CT comparison (axial / coronal / sagittal) across moving / fixed / warped."""
    titles = ["Moving (phase 50)", f"Fixed (phase {phase})", f"Warped (50→{phase})", "|Fixed − Warped|"]
    axes_labels = ["Axial mid", "Coronal mid", "Sagittal mid"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    diff = np.abs(fixed - warped)
    for r, ax_axis in enumerate([0, 1, 2]):
        panels = [_mid(moving, ax_axis), _mid(fixed, ax_axis), _mid(warped, ax_axis), _mid(diff, ax_axis)]
        for c, panel in enumerate(panels):
            cmap = "magma" if c == 3 else "gray"
            axes[r, c].imshow(panel, cmap=cmap)
            if r == 0:
                axes[r, c].set_title(titles[c])
            if c == 0:
                axes[r, c].set_ylabel(axes_labels[r])
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.tight_layout()
    return fig


def fig_jacobian(jacobian: np.ndarray):
    """Jacobian determinant: axial / coronal / sagittal mid-slices + histogram."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, axis, title in zip(axes[:3], [0, 1, 2], ["Axial", "Coronal", "Sagittal"]):
        im = ax.imshow(_mid(jacobian, axis), cmap="coolwarm", vmin=0.0, vmax=2.0)
        ax.set_title(f"Jacobian {title} mid")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].hist(jacobian.ravel(), bins=120, range=(-0.5, 3.0), color="tab:blue", alpha=0.8)
    axes[3].axvline(0.0, color="red", lw=1.0, ls="--", label="J=0 (fold)")
    axes[3].axvline(1.0, color="green", lw=0.5, ls=":", label="J=1 (identity)")
    axes[3].set_xlabel("Jacobian"); axes[3].set_ylabel("voxel count")
    axes[3].set_title("Jacobian histogram"); axes[3].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_dvf_magnitude(dvf: np.ndarray):
    mag = np.linalg.norm(dvf, axis=-1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmax = float(np.percentile(mag, 99.0))
    for ax, axis, title in zip(axes, [0, 1, 2], ["Axial", "Coronal", "Sagittal"]):
        im = ax.imshow(_mid(mag, axis), cmap="viridis", vmin=0.0, vmax=max(vmax, 1.0))
        ax.set_title(f"|u| {title} mid")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[-1], fraction=0.046, label="mm")
    fig.tight_layout()
    return fig


def fig_folding(jacobian: np.ndarray):
    folding = (jacobian <= 0).astype(np.uint8)
    if folding.sum() == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No folding voxels (J > 0 everywhere)", ha="center", va="center",
                fontsize=13, color="green", fontweight="bold")
        return fig
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, axis, title in zip(axes, [0, 1, 2], ["Axial", "Coronal", "Sagittal"]):
        ax.imshow(_mid(folding, axis), cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"Folding mask {title} mid")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def fig_metrics_table(metrics: dict, phase: str):
    rows = _select_metric_rows(metrics)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.95, f"DVF QA metrics — phase {phase}", ha="center", va="top",
            fontsize=18, fontweight="bold")
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Metric", "Value"],
        cellLoc="left", colLoc="left", loc="center", colWidths=[0.55, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    return fig


def _select_metric_rows(metrics: dict) -> list[tuple[str, str]]:
    keys = [
        "qa_status",
        "jac_min", "jac_p01", "jac_p05", "jac_median", "jac_mean",
        "jac_p95", "jac_p99", "jac_max",
        "fraction_jac_le_0", "fraction_jac_lt_0_2", "fraction_jac_gt_5",
        "folding_component_count", "largest_folding_component_ml",
        "dvf_magnitude_p95_mm", "dvf_magnitude_max_mm",
        "gradient_norm_mean", "bending_energy_mean",
        "warped_vs_fixed_mse", "warped_vs_fixed_mae", "warped_vs_fixed_ncc",
        "our_warp_vs_precomputed_mse", "our_warp_vs_precomputed_mae", "our_warp_vs_precomputed_ncc",
    ]
    rows: list[tuple[str, str]] = []
    for k in keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                rows.append((k, f"{v:.6g}"))
            else:
                rows.append((k, str(v)))
    return rows


# ---------- report writers ----------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_html(path: Path, phase: str, figs: dict, metrics: dict) -> None:
    sections: list[str] = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(
        f"<header><h1>DVF QA — phase {html_mod.escape(phase)}</h1>"
        f"<p><b>Generated:</b> {html_mod.escape(now)}</p>"
        f"<p><b>qa_status:</b> <code>{html_mod.escape(str(metrics.get('qa_status','?')))}</code></p></header>"
    )
    for title, fig in figs.items():
        b64 = _fig_to_b64(fig)
        sections.append(
            f"<section><h2>{html_mod.escape(title)}</h2>"
            f"<img src='data:image/png;base64,{b64}'/></section>"
        )
    rows = "".join(
        f"<tr><th>{html_mod.escape(k)}</th><td>{html_mod.escape(v)}</td></tr>"
        for k, v in _select_metric_rows(metrics)
    )
    sections.append(f"<section><h2>Metrics</h2><table>{rows}</table></section>")

    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<title>DVF QA — phase {html_mod.escape(phase)}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }}
h1 {{ margin: 0 0 8px 0; }} h2 {{ margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
table {{ border-collapse: collapse; margin: 8px 0 16px 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
th {{ background: #f6f6f6; }} img {{ max-width: 100%; height: auto; }}
section {{ margin-bottom: 16px; }}
</style></head><body>
{chr(10).join(sections)}
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_pdf(path: Path, phase: str, figs: dict, metrics: dict) -> None:
    with PdfPages(path) as pdf:
        # Cover page with metrics table
        cover = fig_metrics_table(metrics, phase)
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        # Each figure as one page (recreated since previous savefig closed it)
        for title, fig in figs.items():
            fig.suptitle(title, fontsize=12, y=1.02)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        meta = pdf.infodict()
        meta["Title"] = f"DVF QA — phase {phase}"
        meta["Author"] = "dvf-qa"
        meta["CreationDate"] = dt.datetime.now()


# ---------- per-phase orchestration ----------

def run_phase(phase: str, root: Path, out_dir: Path, *, moving: np.ndarray, edge_crop: int) -> dict:
    print(f"\n=== phase {phase} ===")
    dvf, spacing = load_dvf(root / "DVF" / f"prediction_{phase}.nii.gz")
    fixed = load_4d_phase(root / "4D", phase)
    print(f"  shapes: dvf={dvf.shape} moving={moving.shape} fixed={fixed.shape} spacing={spacing}")

    interior_mask = make_interior_mask(dvf.shape[:3], crop=edge_crop)
    dvf_metrics, jacobian = summarize_dvf_qa(dvf, spacing, mask=interior_mask)
    warped = warp_volume_moving_to_fixed(moving, dvf, spacing).astype(np.float32)
    sim = image_similarity(warped, fixed, mask=interior_mask, prefix="warped_vs_fixed_")

    cross_sim: dict[str, float | None] = {}
    precomp_path = root / "Other" / f"deformed_image_50to{phase}.nii.gz"
    if precomp_path.is_file():
        precomp = load_nifti(precomp_path)
        cross_sim = image_similarity(warped, precomp, mask=interior_mask,
                                     prefix="our_warp_vs_precomputed_")

    metrics: dict = {"phase": phase, **dvf_metrics, **sim, **cross_sim}
    metrics["ct_shape"] = list(moving.shape)
    metrics["dvf_spacing"] = list(spacing)
    metrics["edge_crop_voxels"] = int(edge_crop)

    phase_out = out_dir / f"phase_{phase}"
    phase_out.mkdir(parents=True, exist_ok=True)
    (phase_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Build *separate* figure objects per output (PDF and HTML each call _fig_to_b64/savefig which close fig)
    figs_for_html = {
        "CT comparison (moving / fixed / warped / |diff|)": fig_ct_comparison(moving, fixed, warped, phase),
        "Jacobian determinant + histogram":                  fig_jacobian(jacobian),
        "DVF magnitude":                                     fig_dvf_magnitude(dvf),
        "Folding voxels (J ≤ 0)":                            fig_folding(jacobian),
    }
    write_html(phase_out / "report.html", phase, figs_for_html, metrics)

    figs_for_pdf = {
        "CT comparison (moving / fixed / warped / |diff|)": fig_ct_comparison(moving, fixed, warped, phase),
        "Jacobian determinant + histogram":                  fig_jacobian(jacobian),
        "DVF magnitude":                                     fig_dvf_magnitude(dvf),
        "Folding voxels (J ≤ 0)":                            fig_folding(jacobian),
    }
    write_pdf(phase_out / "report.pdf", phase, figs_for_pdf, metrics)

    ncc = metrics.get("warped_vs_fixed_ncc")
    print(f"  qa_status={metrics.get('qa_status')}  warped_vs_fixed_ncc={ncc:.4f}"
          if isinstance(ncc, float) else f"  qa_status={metrics.get('qa_status')}")
    return metrics


def write_top_index(out_dir: Path, results: list[dict]) -> None:
    rows = []
    for r in results:
        phase = r["phase"]
        ncc = r.get("warped_vs_fixed_ncc")
        ncc_s = f"{ncc:.4f}" if isinstance(ncc, float) else "n/a"
        cross = r.get("our_warp_vs_precomputed_ncc")
        cross_s = f"{cross:.4f}" if isinstance(cross, float) else "n/a"
        status = r.get("qa_status", "?")
        cls = {"PASS": "good", "WARNING": "warn", "FAIL": "fail"}.get(status, "")
        rows.append(
            f"<tr><td><a href='phase_{phase}/report.html'>phase {phase}</a></td>"
            f"<td><a href='phase_{phase}/report.pdf'>PDF</a></td>"
            f"<td class='{cls}'>{status}</td>"
            f"<td>{r.get('jac_min', float('nan')):.3f}</td>"
            f"<td>{r.get('fraction_jac_le_0', float('nan')):.3e}</td>"
            f"<td>{r.get('largest_folding_component_ml', float('nan')):.3f}</td>"
            f"<td>{r.get('dvf_magnitude_p95_mm', float('nan')):.3f}</td>"
            f"<td>{ncc_s}</td><td>{cross_s}</td></tr>"
        )
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>DVF predict QA</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #eee; }}
.good {{ background: #d4f5d4; }} .warn {{ background: #fff3cd; }} .fail {{ background: #fbd1d1; }}
</style></head>
<body>
<h1>DVF predict QA</h1>
<p>Predicted DVFs from phase 50 to each other phase. PASS = no folding voxels and within drift thresholds.</p>
<table>
<tr><th>Phase</th><th>PDF</th><th>QA</th><th>Jac min</th>
<th>frac J≤0</th><th>largest fold (mL)</th>
<th>|u| p95 (mm)</th><th>NCC (warped vs fixed)</th>
<th>NCC (vs precomputed)</th></tr>
{chr(10).join(rows)}
</table>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="DVF_predict directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phases", nargs="*", default=PHASES,
                        help="Subset of phases to process (default: all 9)")
    parser.add_argument("--edge-crop", type=int, default=3,
                        help="Voxels to exclude from each side of the volume in QA stats")
    args = parser.parse_args()

    if not (args.root / "DVF").is_dir():
        print(f"Expected {args.root}/DVF/ with prediction_PP.nii.gz files", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading moving CT (phase 50) from 3D/ ...")
    moving = read_dicom_dir(args.root / "3D")
    print(f"  moving shape={moving.shape}")

    results: list[dict] = []
    for phase in args.phases:
        try:
            results.append(run_phase(phase, args.root, args.out_dir,
                                     moving=moving, edge_crop=args.edge_crop))
        except Exception:
            print(f"FAILED phase {phase}:")
            traceback.print_exc()
    if results:
        write_top_index(args.out_dir, results)
        print(f"\nIndex: {args.out_dir / 'index.html'}")
        print(f"Summary: {args.out_dir / 'summary.json'}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
