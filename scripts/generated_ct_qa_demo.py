"""Run generated_ct_qa modules on DVF_predict/ data and emit unified reports.

The generated_ct_qa subtree expects:
- synthetic_ct: 3D HU array of the AI-generated CT (we use ``warp(phase50, predicted_DVF)``)
- planning_ct:  3D HU array of the reference CT (we use ``DVF_predict/3D/`` = phase 50)
- input_image:  2D X-ray-like image (we use ``DVF_predict/AP/AP_PP.png``)
- 4d_ct:        list of 3D arrays representing temporal phases (we use warped CTs across phases)
- (optional dose / structure masks — not available for DVF_predict, that module returns ERROR)

For each target phase (00, 10, ..., 90 except 50) we call each QA module
directly, then write per-phase PDF + HTML + metrics.json plus a top-level
``index.html`` / ``summary.json`` under ``outputs/generated_ct_qa/``.

PDF / HTML are produced via matplotlib PdfPages and base64-embedded PNG to
stay consistent with the rest of the pipeline reports (no extra dependency
on ``reportlab``).
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

# Make generated_ct_qa importable
_GEN_SRC = Path(__file__).resolve().parents[1] / "generated_ct_qa" / "src"
sys.path.insert(0, str(_GEN_SRC))

from qa_system.config_manager import ConfigManager  # noqa: E402
from qa_system.dosimetric_accuracy import DosimetricAccuracy  # noqa: E402
from qa_system.geom_integrity import GeomIntegrity  # noqa: E402
from qa_system.robustness_check import RobustnessCheck  # noqa: E402
from qa_system.temporal_motion import TemporalMotion  # noqa: E402

from dvf_qa.warp import warp_volume_moving_to_fixed  # noqa: E402


PHASES = ["00", "10", "20", "30", "40", "60", "70", "80", "90"]


# ---------- I/O ----------

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


def load_dvf(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = tuple(float(s) for s in img.GetSpacing()[:3])
    return arr, spacing


def load_input_image(path: Path) -> np.ndarray:
    import imageio.v3 as iio
    img = iio.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img


# ---------- per-phase QA ----------

def run_temporal_once(planning: np.ndarray, all_phase_warped: dict[str, np.ndarray]) -> dict:
    """TemporalMotion is the expensive one (DIR per consecutive pair).

    The 4d_ct input is identical across per-phase reports, so we compute it
    once here and reuse the result. Cuts wall-clock from ~9× to ~1×.
    """
    phases_ordered = ["00", "10", "20", "30", "40", "50", "60", "70", "80", "90"]
    fourd = [planning if p == "50" else all_phase_warped[p] for p in phases_ordered]
    print("\n--- TemporalMotion (computed once, shared by all phases) ---")
    module = TemporalMotion()
    try:
        results = module.validate({"4d_ct": [v.astype(np.float64) for v in fourd]})
        status = module.status if not (isinstance(results, dict) and results.get("status") == "ERROR") else "ERROR"
        out = {"status": status, "results": results}
    except Exception as exc:
        out = {"status": "ERROR", "results": {"error": str(exc)}}
    print(f"  TemporalMotion: {out['status']}")
    return out


def run_phase_qa(phase: str, root: Path, *, moving: np.ndarray,
                 all_phase_warped: dict[str, np.ndarray],
                 temporal_shared: dict) -> dict:
    print(f"\n=== phase {phase} ===")
    synthetic = all_phase_warped[phase]
    planning = moving

    ap_png = root / "AP" / f"AP_{phase}.png"
    input_image = load_input_image(ap_png) if ap_png.is_file() else None

    data = {
        "synthetic_ct": synthetic.astype(np.float64),
        "planning_ct": planning.astype(np.float64),
        "input_image": input_image,
        "voxel_size": (1.0, 1.0, 1.0),
    }

    modules_results: dict[str, dict] = {}
    # TemporalMotion shared
    modules_results["TemporalMotion"] = temporal_shared

    # Per-phase modules
    for cls in (RobustnessCheck, GeomIntegrity, DosimetricAccuracy):
        module = cls()
        try:
            results = module.validate(data)
            status = module.status if not (isinstance(results, dict) and results.get("status") == "ERROR") else "ERROR"
            modules_results[module.name] = {"status": status, "results": results}
        except Exception as exc:
            modules_results[module.name] = {"status": "ERROR", "results": {"error": str(exc)}}
        print(f"  {module.name}: {modules_results[module.name]['status']}")
    return {"phase": phase, "modules": modules_results}


# ---------- report ----------

def _mid(volume: np.ndarray, axis: int) -> np.ndarray:
    sl = np.take(volume, volume.shape[axis] // 2, axis=axis)
    if axis in (1, 2):
        sl = np.flipud(sl)
    return sl


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fig_ct_panel(synthetic: np.ndarray, planning: np.ndarray, phase: str):
    diff = np.abs(synthetic - planning)
    titles = ["Synthetic (warped 50→PP)", "Planning (phase 50)", "|Synthetic − Planning|"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes_labels = ["Axial mid", "Coronal mid", "Sagittal mid"]
    for r, ax_axis in enumerate([0, 1, 2]):
        panels = [_mid(synthetic, ax_axis), _mid(planning, ax_axis), _mid(diff, ax_axis)]
        for c, panel in enumerate(panels):
            cmap = "magma" if c == 2 else "gray"
            axes[r, c].imshow(panel, cmap=cmap)
            if r == 0:
                axes[r, c].set_title(titles[c])
            if c == 0:
                axes[r, c].set_ylabel(axes_labels[r])
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle(f"phase {phase} — CT comparison", fontsize=13)
    fig.tight_layout()
    return fig


def fig_modules_summary(phase_result: dict):
    mods = phase_result["modules"]
    rows: list[tuple[str, str]] = []
    for mod_name, mod_res in mods.items():
        rows.append((mod_name, mod_res.get("status", "?")))
    fig, ax = plt.subplots(figsize=(8.27, 4))
    ax.axis("off")
    ax.text(0.5, 0.95, f"phase {phase_result['phase']} — Module status",
            ha="center", va="top", fontsize=16, fontweight="bold")
    colors = []
    for _, status in rows:
        c = "#d4f5d4" if status == "PASS" else ("#fff3cd" if status == "WARNING" else "#fbd1d1")
        colors.append([c, c])
    table = ax.table(cellText=rows, colLabels=["Module", "Status"],
                     cellLoc="left", colLoc="left", loc="center",
                     colWidths=[0.5, 0.3], cellColours=colors)
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 1.6)
    return fig


def _flatten_for_table(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for k, v in d.items():
        if k.startswith("plot_"):
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            rows.extend(_flatten_for_table(v, key + "_"))
        elif isinstance(v, list):
            rows.append((key, f"list[{len(v)}]"))
        elif isinstance(v, float):
            rows.append((key, f"{v:.5g}"))
        else:
            rows.append((key, str(v)))
    return rows


def fig_module_details(mod_name: str, mod_res: dict):
    rows = _flatten_for_table(mod_res.get("results", {}) or {})
    if not rows:
        rows = [("(no metrics)", "")]
    rows = rows[:40]  # cap for layout
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.98, f"{mod_name} — details ({mod_res.get('status','?')})",
            ha="center", va="top", fontsize=14, fontweight="bold")
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     cellLoc="left", colLoc="left", loc="center",
                     colWidths=[0.55, 0.4])
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 1.4)
    return fig


def fig_attached_plot(plot_path: str, title: str):
    if not Path(plot_path).is_file():
        return None
    fig, ax = plt.subplots(figsize=(8.27, 6))
    img = plt.imread(plot_path)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def collect_figures(phase_result: dict, synthetic: np.ndarray, planning: np.ndarray) -> dict:
    figs: dict = {}
    figs["Module status"] = fig_modules_summary(phase_result)
    figs["CT comparison"] = fig_ct_panel(synthetic, planning, phase_result["phase"])
    for mod_name, mod_res in phase_result["modules"].items():
        figs[f"{mod_name} — details"] = fig_module_details(mod_name, mod_res)
        for k, v in (mod_res.get("results") or {}).items():
            if isinstance(v, str) and k.startswith("plot_") and Path(v).is_file():
                fig = fig_attached_plot(v, f"{mod_name}: {k}")
                if fig is not None:
                    figs[f"{mod_name} — {k}"] = fig
    return figs


def write_pdf(path: Path, phase_result: dict, figs: dict) -> None:
    with PdfPages(path) as pdf:
        for title, fig in figs.items():
            fig.suptitle(title, fontsize=12, y=1.02)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = f"Generated CT QA — phase {phase_result['phase']}"
        info["Author"] = "lung-4d-pipeline"
        info["CreationDate"] = dt.datetime.now()


def write_html(path: Path, phase_result: dict, figs: dict) -> None:
    sections: list[str] = []
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phase = phase_result["phase"]
    sections.append(f"<header><h1>Generated CT QA — phase {html_mod.escape(phase)}</h1>"
                    f"<p><b>Generated:</b> {html_mod.escape(now)}</p></header>")
    for title, fig in figs.items():
        b64 = _fig_to_b64(fig)
        sections.append(f"<section><h2>{html_mod.escape(title)}</h2>"
                        f"<img src='data:image/png;base64,{b64}'/></section>")
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<title>Generated CT QA — phase {html_mod.escape(phase)}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-top: 18px; }} h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
img {{ max-width: 100%; height: auto; }}
section {{ margin-bottom: 16px; }}
</style></head><body>
{chr(10).join(sections)}
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_top_index(out_dir: Path, results: list[dict]) -> None:
    rows = []
    for r in results:
        phase = r["phase"]
        # build "PASS / FAIL / ERROR" badge per module
        cells = []
        for mod_name in ("RobustnessCheck", "GeomIntegrity", "TemporalMotion", "DosimetricAccuracy"):
            status = r["modules"].get(mod_name, {}).get("status", "?")
            cls = {"PASS": "good", "FAIL": "fail", "ERROR": "err", "WARNING": "warn"}.get(status, "")
            cells.append(f"<td class='{cls}'>{status}</td>")
        rows.append(
            f"<tr><td><a href='phase_{phase}/report.html'>phase {phase}</a></td>"
            f"<td><a href='phase_{phase}/report.pdf'>PDF</a></td>"
            f"{''.join(cells)}</tr>"
        )
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>Generated CT QA</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
th {{ background: #eee; }}
.good {{ background: #d4f5d4; }} .warn {{ background: #fff3cd; }}
.fail {{ background: #fbd1d1; }} .err {{ background: #e7daff; }}
</style></head>
<body>
<h1>Generated CT QA — DVF_predict cases</h1>
<p>Synthetic CTs created by warping phase 50 with each predicted DVF (50→PP). Dose / structure masks are not available, so DosimetricAccuracy returns ERROR as expected.</p>
<table>
<tr><th>Phase</th><th>PDF</th><th>RobustnessCheck</th><th>GeomIntegrity</th><th>TemporalMotion</th><th>DosimetricAccuracy</th></tr>
{chr(10).join(rows)}
</table>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(results, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="DVF_predict directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phases", nargs="*", default=PHASES)
    args = parser.parse_args()

    # Lock ConfigManager singleton onto the bundled config
    config_path = _GEN_SRC.parent / "config" / "thresholds.json"
    ConfigManager(config_path=str(config_path))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading moving CT (phase 50) ...")
    moving = read_dicom_dir(args.root / "3D")
    print(f"  shape={moving.shape}")

    # Always pre-warp every phase so 4d_ct can be assembled even when only a
    # subset is requested for per-phase reporting (smoke tests etc.).
    print("Pre-warping all phases (50 -> PP) ...")
    all_phase_warped: dict[str, np.ndarray] = {}
    for phase in PHASES:
        dvf, spacing = load_dvf(args.root / "DVF" / f"prediction_{phase}.nii.gz")
        all_phase_warped[phase] = warp_volume_moving_to_fixed(moving, dvf, spacing).astype(np.float32)
    print(f"  warped {len(all_phase_warped)} phases")

    temporal_shared = run_temporal_once(moving, all_phase_warped)

    results: list[dict] = []
    for phase in args.phases:
        try:
            phase_result = run_phase_qa(phase, args.root, moving=moving,
                                        all_phase_warped=all_phase_warped,
                                        temporal_shared=temporal_shared)
            phase_out = args.out_dir / f"phase_{phase}"
            phase_out.mkdir(parents=True, exist_ok=True)
            (phase_out / "metrics.json").write_text(
                json.dumps(phase_result, indent=2,
                           default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)),
                encoding="utf-8",
            )
            figs_html = collect_figures(phase_result, all_phase_warped[phase], moving)
            write_html(phase_out / "report.html", phase_result, figs_html)
            figs_pdf = collect_figures(phase_result, all_phase_warped[phase], moving)
            write_pdf(phase_out / "report.pdf", phase_result, figs_pdf)
            results.append(phase_result)
        except Exception:
            print(f"FAILED phase {phase}:")
            traceback.print_exc()

    if results:
        write_top_index(args.out_dir, results)
        print(f"\nIndex: {args.out_dir / 'index.html'}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
