"""Publish a pipeline-run summary to Notion via the official REST API.

Reads one or more ``outputs/*_demo*/summary.json`` files and appends a row per
case to a Notion database (one row = one case run). Optionally creates a
top-level page that links back to the GitHub Pages site for browseable
reports.

Setup (one-time, by a human in Notion + GitHub)
-----------------------------------------------
1. In Notion, create an Internal Integration:
   https://www.notion.so/profile/integrations -> "+ New integration"
   Capabilities needed: Read content, Update content, Insert content.
2. Copy the *Internal Integration Token* (starts with ``ntn_...`` or ``secret_...``).
3. Create a Notion database with the following properties:
       - Title (title) — case label
       - Run ID (rich_text)
       - Source (select; values: DIR-Lab, TCIA, etc.)
       - Mean NCC (number)
       - Best NCC (number)
       - AP freq Hz (number)
       - Lat freq Hz (number)
       - Pages URL (url)
       - Date (date)
4. Share the database with the integration:
       Open the database -> ``...`` menu -> ``Add connections`` -> your integration.
5. Copy the database id (32-char hex from the URL).
6. Add GitHub secrets ``NOTION_API_KEY`` and ``NOTION_DATABASE_ID``.
   For local runs export them as env vars.

Usage::

    NOTION_API_KEY=... NOTION_DATABASE_ID=... python scripts/notion_publish.py \\
        --summary outputs/dirlab_demo/summary.json \\
        --source DIR-Lab \\
        --pages-base-url https://<org>.github.io/dvf-qa-reports/runs/<run-id>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


NOTION_API_VERSION = "2022-06-28"
NOTION_API_BASE = "https://api.notion.com/v1"


def post(path: str, body: dict, token: str) -> dict:
    req = urllib.request.Request(
        f"{NOTION_API_BASE}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Notion API {e.code}: {msg}") from None


def case_to_page(row: dict, *, database_id: str, source: str, run_id: str, pages_base_url: str | None) -> dict:
    """Dispatch on row schema: respiratory cycle pairing vs. DVF QA."""
    if "phase" in row and "case_id" not in row:
        return _dvf_row_to_page(row, database_id=database_id, source=source,
                                run_id=run_id, pages_base_url=pages_base_url)
    return _respiratory_row_to_page(row, database_id=database_id, source=source,
                                    run_id=run_id, pages_base_url=pages_base_url)


def _respiratory_row_to_page(row: dict, *, database_id: str, source: str,
                              run_id: str, pages_base_url: str | None) -> dict:
    case_id = row.get("case_id") or row.get("case") or "unknown"
    ncc = row.get("best_pair_ncc")
    ap_hz = row.get("ap_dominant_frequency_hz")
    lat_hz = row.get("lateral_dominant_frequency_hz")
    pages_url = (
        f"{pages_base_url.rstrip('/')}/{case_id.split('-')[-1]}/report.html"
        if pages_base_url else None
    )
    today = dt.date.today().isoformat()

    properties: dict = {
        "Title": {"title": [{"text": {"content": str(case_id)}}]},
        "Run ID": {"rich_text": [{"text": {"content": run_id}}]},
        "Source": {"select": {"name": source}},
        "Date": {"date": {"start": today}},
    }
    if isinstance(ncc, (int, float)):
        properties["Best NCC"] = {"number": float(ncc)}
    if isinstance(ap_hz, (int, float)):
        properties["AP freq Hz"] = {"number": float(ap_hz)}
    if isinstance(lat_hz, (int, float)):
        properties["Lat freq Hz"] = {"number": float(lat_hz)}
    if pages_url:
        properties["Pages URL"] = {"url": pages_url}
    return {"parent": {"database_id": database_id}, "properties": properties}


def _dvf_row_to_page(row: dict, *, database_id: str, source: str,
                     run_id: str, pages_base_url: str | None) -> dict:
    phase = str(row.get("phase", "?"))
    qa_status = row.get("qa_status", "?")
    title = f"DVF-QA phase {phase} ({qa_status})"
    ncc = row.get("warped_vs_fixed_ncc")
    pages_url = (
        f"{pages_base_url.rstrip('/')}/phase_{phase}/report.html"
        if pages_base_url else None
    )
    today = dt.date.today().isoformat()

    properties: dict = {
        "Title": {"title": [{"text": {"content": title}}]},
        "Run ID": {"rich_text": [{"text": {"content": run_id}}]},
        "Source": {"select": {"name": source}},
        "Date": {"date": {"start": today}},
    }
    if isinstance(ncc, (int, float)):
        properties["Best NCC"] = {"number": float(ncc)}
    if pages_url:
        properties["Pages URL"] = {"url": pages_url}
    return {"parent": {"database_id": database_id}, "properties": properties}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path, help="Path to outputs/<run>/summary.json")
    parser.add_argument("--source", default="DIR-Lab", help="Source tag used in Notion DB")
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--pages-base-url", help="GitHub Pages base URL for this run (e.g. .../runs/<run-id>)")
    parser.add_argument("--limit", type=int, help="Optional cap on number of rows uploaded (smoke testing)")
    args = parser.parse_args()

    token = os.environ.get("NOTION_API_KEY")
    db_id = os.environ.get("NOTION_DATABASE_ID")
    if not token or not db_id:
        print("NOTION_API_KEY and NOTION_DATABASE_ID env vars are required", file=sys.stderr)
        return 2
    if not args.summary.is_file():
        print(f"summary not found: {args.summary}", file=sys.stderr)
        return 2

    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    created = []
    for row in rows:
        body = case_to_page(row, database_id=db_id, source=args.source,
                            run_id=args.run_id, pages_base_url=args.pages_base_url)
        res = post("/pages", body, token)
        label = row.get("case_id") or (f"phase {row['phase']}" if "phase" in row else "?")
        created.append({"id": res.get("id"), "label": label})
        print(f"  uploaded {label} -> {res.get('id')}")
    print(f"\nUploaded {len(created)} rows to Notion database {db_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
