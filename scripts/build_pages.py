"""Aggregate pipeline run outputs into a static site ready for GitHub Pages.

Collects every ``outputs/*_demo*/`` directory that contains an ``index.html``
or a ``summary.json``, stamps it with a run id, and writes them under
``_site/runs/<run_id>/``. Generates a top-level ``_site/index.html`` listing
all known runs with the headline metrics. The result is a fully self-contained
static site (no external assets) suitable for GitHub Pages, Cloudflare Pages,
or any static host.

Usage::

    python scripts/build_pages.py --site-dir _site

Run id defaults to ``<source-dirname>_<modified-date>``; pass ``--run-id`` to
override (e.g. for re-deploying a fixed snapshot).
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import shutil
import sys
from pathlib import Path


RAW_DATA_DIRS = {"dirlab_data", "tcia_4d_lung_data", "popi_data"}


def find_runs(repo_root: Path, *, include_all: bool = False) -> list[Path]:
    """Return candidate output directories that look like reportable runs.

    By default this is *opt-in*: only directories that ship a top-level
    ``summary.json`` (created by the orchestrator scripts) are included.
    Pass ``include_all=True`` to also surface ad-hoc directories that have
    an ``index.html`` / ``report.html`` / per-case ``metrics.json``.

    Raw-data directories listed in :data:`RAW_DATA_DIRS` are always skipped.
    """
    out_root = repo_root / "outputs"
    if not out_root.is_dir():
        return []
    runs: list[Path] = []
    for child in sorted(out_root.iterdir()):
        if not child.is_dir() or child.name in RAW_DATA_DIRS:
            continue
        if (child / "summary.json").is_file():
            runs.append(child)
            continue
        if not include_all:
            continue
        if (child / "index.html").is_file():
            runs.append(child)
            continue
        if (child / "report.html").is_file() or (child / "report.pdf").is_file():
            runs.append(child)
            continue
        if any(p.is_file() for p in child.glob("*/metrics.json")):
            runs.append(child)
    return runs


def load_summary(run_dir: Path) -> tuple[list[dict], dict]:
    """Return per-case rows + headline metrics for a run directory."""
    sj = run_dir / "summary.json"
    rows: list[dict] = []
    if sj.is_file():
        try:
            rows = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    if not rows:
        # Fall back to per-case metrics.json files
        for case in sorted(run_dir.iterdir()):
            mj = case / "metrics.json"
            if mj.is_file():
                try:
                    rows.append(json.loads(mj.read_text(encoding="utf-8")))
                except Exception:
                    pass

    nccs = [r["best_pair_ncc"] for r in rows if isinstance(r.get("best_pair_ncc"), (int, float))]
    headline: dict[str, str | float | int] = {
        "cases": len(rows),
        "mean_ncc": sum(nccs) / len(nccs) if nccs else float("nan"),
        "min_ncc": min(nccs) if nccs else float("nan"),
        "max_ncc": max(nccs) if nccs else float("nan"),
        "success_count_0_9": sum(1 for n in nccs if n >= 0.9),
    }
    return rows, headline


def copy_run(run_dir: Path, site_dir: Path, run_id: str) -> Path:
    target = site_dir / "runs" / run_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run_dir, target)
    return target


def write_top_index(site_dir: Path, manifest: list[dict]) -> None:
    rows_html: list[str] = []
    for entry in manifest:
        m = entry["headline"]
        mean = f"{m['mean_ncc']:.3f}" if isinstance(m["mean_ncc"], float) and m["mean_ncc"] == m["mean_ncc"] else "n/a"
        minimum = f"{m['min_ncc']:.3f}" if isinstance(m["min_ncc"], float) and m["min_ncc"] == m["min_ncc"] else "n/a"
        link = f"runs/{entry['run_id']}/index.html"
        rows_html.append(
            f"<tr><td><a href='{html.escape(link)}'>{html.escape(entry['run_id'])}</a></td>"
            f"<td>{html.escape(entry['source'])}</td>"
            f"<td>{m['cases']}</td>"
            f"<td>{mean}</td>"
            f"<td>{minimum}</td>"
            f"<td>{m['success_count_0_9']}/{m['cases']}</td>"
            f"<td>{html.escape(entry['captured'])}</td></tr>"
        )
    body = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<title>DVF QA — pipeline runs</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
th {{ background: #f3f3f3; }}
.note {{ color: #555; font-size: 14px; }}
</style></head>
<body>
<h1>DVF QA — pipeline runs</h1>
<p class="note">Each row links to the per-run report index. Reports include
full per-case PDF + HTML + metrics.json. This page is meant for the project
team; please do not redistribute the underlying derived images of DIR-Lab
data outside the team without explicit permission.</p>
<table>
<tr><th>Run</th><th>Source</th><th>Cases</th><th>Mean NCC</th><th>Min NCC</th><th>Success @ NCC&ge;0.9</th><th>Captured</th></tr>
{chr(10).join(rows_html)}
</table>
</body></html>"""
    (site_dir / "index.html").write_text(body, encoding="utf-8")
    (site_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--site-dir", type=Path, default=Path("_site"))
    parser.add_argument("--run-id", help="Force a single run id; only meaningful with --only")
    parser.add_argument("--only", nargs="*", help="Restrict to specific outputs/* dirnames")
    parser.add_argument("--include-all", action="store_true",
                        help="Surface all output dirs (not just those with summary.json)")
    args = parser.parse_args()

    repo_root = args.repo_root
    site_dir = args.site_dir
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "runs").mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")

    runs = find_runs(repo_root, include_all=args.include_all)
    if args.only:
        chosen = set(args.only)
        runs = [r for r in runs if r.name in chosen]
    if not runs:
        print("No demo runs found under outputs/", file=sys.stderr)
        return 1

    manifest: list[dict] = []
    for run_dir in runs:
        rows, headline = load_summary(run_dir)
        modified = dt.datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        # Stable run id: directory name only. Including the mtime made the id
        # diverge between the build job (which deploys to Pages) and the
        # notify-slack job (which composes links) because each does its own
        # git checkout and gets fresh mtimes.
        run_id = args.run_id if (args.run_id and args.only and len(args.only) == 1) else run_dir.name
        copy_run(run_dir, site_dir, run_id)
        manifest.append({
            "run_id": run_id,
            "source": run_dir.name,
            "captured": modified,
            "headline": headline,
            "n_rows": len(rows),
        })
        print(f"staged {run_dir.name} -> runs/{run_id}/  (cases={len(rows)}, mean NCC={headline['mean_ncc']:.3f})")

    manifest.sort(key=lambda e: e["captured"], reverse=True)
    write_top_index(site_dir, manifest)
    print(f"\nBuilt site at {site_dir.resolve()}")
    print(f"  Top index: {site_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
