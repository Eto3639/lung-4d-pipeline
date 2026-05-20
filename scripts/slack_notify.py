"""Post a Slack Block Kit notification summarising one Pages deploy.

Reads ``_site/manifest.json`` (produced by :mod:`scripts.build_pages`),
builds a header + per-run section message with action buttons, and POSTs
to the Slack incoming webhook in ``SLACK_WEBHOOK_URL``.

Run locally::

    SLACK_WEBHOOK_URL=... PAGES_URL=https://.../ \\
        python scripts/slack_notify.py --manifest _site/manifest.json

Designed for the ``notify-slack`` step of
``.github/workflows/deploy-reports.yml``; that workflow sets
``PAGES_URL``, ``GITHUB_SHA``, and ``GITHUB_REF_NAME`` for free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


RUN_LABELS: dict[str, str] = {
    "dirlab_demo": "DIR-Lab 10-case QA",
    "dirlab_lateral_variants": "DIR-Lab lateral strategies",
    "tcia_4d_lung_demo": "TCIA 4D-Lung pilot",
    "tcia_4d_lung_demo_full": "TCIA 4D-Lung 10-phase",
    "dvf_predict_qa": "DVF predict QA",
}


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def summary_text(manifest: list[dict]) -> str:
    if not manifest:
        return "DVF QA reports updated (no runs detected)"
    parts: list[str] = []
    for entry in manifest:
        m = entry["headline"]
        source = entry["source"]
        cases = m.get("cases", 0)
        mean_ncc = m.get("mean_ncc")
        ncc_part = f"meanNCC={mean_ncc:.3f}" if isinstance(mean_ncc, (int, float)) and mean_ncc == mean_ncc else "meanNCC=n/a"
        parts.append(f"{RUN_LABELS.get(source, source)} (cases={cases}, {ncc_part})")
    return "DVF QA reports updated — " + " / ".join(parts)


def build_run_section(entry: dict, pages_url: str) -> dict:
    m = entry["headline"]
    cases = m.get("cases", 0)
    mean_ncc = m.get("mean_ncc")
    min_ncc = m.get("min_ncc")
    success = m.get("success_count_0_9", 0)
    label = RUN_LABELS.get(entry["source"], entry["source"])
    run_url = f"{pages_url.rstrip('/')}/runs/{entry['run_id']}/index.html"

    ncc_text = (
        f"meanNCC=*{mean_ncc:.3f}*  minNCC=*{min_ncc:.3f}*"
        if isinstance(mean_ncc, (int, float)) and mean_ncc == mean_ncc
        else "NCC=n/a (DVF predict QA — Jacobian/folding focus)"
    )
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*<{run_url}|{label}>*\n"
                f"cases={cases}, success@NCC≥0.9 = {success}/{cases}\n"
                f"{ncc_text}\n"
                f"captured: {entry.get('captured', '?')}"
            ),
        },
    }


def build_blocks(manifest: list[dict], *, pages_url: str, commit: str | None, branch: str | None) -> list[dict]:
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": "📊 DVF QA レポート更新"}},
        {"type": "section", "text": {"type": "mrkdwn",
                                      "text": f"トップ index: <{pages_url}|{pages_url}>"}},
        {"type": "divider"},
    ]
    if not manifest:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "_No runs detected in manifest._"}})
    else:
        for entry in manifest:
            blocks.append(build_run_section(entry, pages_url))
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "actions",
        "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "🌐 ライブレポートを開く"},
             "url": pages_url, "style": "primary"},
            {"type": "button", "text": {"type": "plain_text", "text": "📦 リポジトリ"},
             "url": "https://github.com/Eto3639/LungPhaseDetection"},
        ],
    })
    context_parts: list[str] = []
    if branch:
        context_parts.append(f"branch *{branch}*")
    if commit:
        context_parts.append(f"commit `{commit[:7]}`")
    if context_parts:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": " · ".join(context_parts)}],
        })
    return blocks


def post_to_slack(webhook_url: str, text: str, blocks: list[dict]) -> None:
    body = {"text": text, "blocks": blocks, "unfurl_links": False}
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            print(f"Slack {resp.status}: {resp.read().decode('utf-8')}")
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Slack webhook {exc.code}: {msg}") from None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("_site/manifest.json"))
    parser.add_argument("--pages-url", default=os.environ.get("PAGES_URL", ""))
    parser.add_argument("--webhook", default=os.environ.get("SLACK_WEBHOOK_URL", ""))
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    parser.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME", ""))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the JSON payload instead of posting")
    args = parser.parse_args()

    if not args.webhook and not args.dry_run:
        print("SLACK_WEBHOOK_URL not set; skipping.", file=sys.stderr)
        return 0
    if not args.pages_url:
        args.pages_url = "https://eto3639.github.io/LungPhaseDetection/"

    manifest = load_manifest(args.manifest)
    text = summary_text(manifest)
    blocks = build_blocks(manifest, pages_url=args.pages_url,
                          commit=args.commit or None, branch=args.branch or None)

    if args.dry_run:
        print(json.dumps({"text": text, "blocks": blocks}, indent=2, ensure_ascii=False))
        return 0
    post_to_slack(args.webhook, text, blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
