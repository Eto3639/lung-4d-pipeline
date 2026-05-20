"""Create a free-form summary page inside the Notion DB.

Each per-case row uploaded by ``notion_publish.py`` only carries metrics.
For team-readable progress notes (background, decisions, lessons learned),
this script adds one richly formatted page to the same database. The page
title and body are defined in :func:`build_summary_2026_05_20` and similar
factory functions; add a new factory + call from ``main`` per milestone.

Run::

    NOTION_API_KEY=... NOTION_DATABASE_ID=... python scripts/notion_publish_summary.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import date


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
    except Exception as exc:  # urllib.error.HTTPError or URLError
        body_text = getattr(exc, "read", lambda: b"")().decode("utf-8", errors="replace")
        raise SystemExit(f"Notion API error: {exc} {body_text}") from None


# ---------- block helpers ----------

def _text(content: str, *, bold: bool = False, code: bool = False, link: str | None = None) -> dict:
    obj: dict = {"type": "text", "text": {"content": content}}
    if link:
        obj["text"]["link"] = {"url": link}
    if bold or code:
        obj["annotations"] = {}
        if bold:
            obj["annotations"]["bold"] = True
        if code:
            obj["annotations"]["code"] = True
    return obj


def heading_1(s: str) -> dict:
    return {"object": "block", "type": "heading_1", "heading_1": {"rich_text": [_text(s)]}}


def heading_2(s: str) -> dict:
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [_text(s)]}}


def heading_3(s: str) -> dict:
    return {"object": "block", "type": "heading_3", "heading_3": {"rich_text": [_text(s)]}}


def paragraph(*rich_texts: dict) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": list(rich_texts)}}


def bullet(*rich_texts: dict) -> dict:
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": list(rich_texts)}}


def code(language: str, content: str) -> dict:
    return {"object": "block", "type": "code", "code": {
        "rich_text": [_text(content)], "language": language,
    }}


def divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def callout(emoji: str, s: str) -> dict:
    return {"object": "block", "type": "callout", "callout": {
        "rich_text": [_text(s)], "icon": {"type": "emoji", "emoji": emoji},
    }}


# ---------- content for 2026-05-20 milestone ----------

PAGES_URL = "https://eto3639.github.io/LungPhaseDetection/"
REPO_URL = "https://github.com/Eto3639/LungPhaseDetection"


def build_summary_2026_05_20() -> tuple[str, list[dict]]:
    title = "DVF QA パイプライン構築まとめ (2026-05-20)"
    blocks: list[dict] = [
        callout("📝",
                "2026-05-19〜20 の作業まとめ。古典的呼吸信号抽出パイプラインを構築し、TCIA + DIR-Lab で検証、"
                "結果を GitHub Pages / Notion に自動配信する基盤まで整備した。"),

        heading_1("背景と目的"),
        paragraph(_text(
            "下流の 2D-4D 再構成モデルの入力として、正面 (AP) と側面 (Lateral) の動態胸部画像から "
            "「最も相関のある1呼吸分のペア」を抽出する必要があった。")),
        paragraph(
            _text("当初は "), _text("PyTorch + U-Net", code=True),
            _text(" による AI 肺セグメンテーションを使う想定だったが、薬事の制約により学習済み AI モデルは使用不可。"
                   "古典的画像処理のみで完結する手法へ全面的に切替えた。"),
        ),

        heading_1("実装した古典手法"),

        heading_2("1. Amsterdam Shroud 法 (呼吸信号抽出)"),
        paragraph(_text(
            "Zijp, Sonke, van Herk (2004) 由来。学習モデル不使用。古い手法だが放射線治療装置等で実績あり。")),
        bullet(_text("各フレームの cranio-caudal (CC) 方向の輝度勾配を計算")),
        bullet(_text("横方向に積算して1次元プロファイル化")),
        bullet(_text("時系列に並べて2次元シュラウド画像 (CC × Time) を生成")),
        bullet(_text("各時刻 column の argmax を放物線フィットでサブピクセル追跡 → 横隔膜の上下動")),
        bullet(_text("帯域通過 (0.1–0.6 Hz) で心拍 (≈1 Hz) と低周波ドリフトを除去")),
        bullet(_text("2次多項式 detrend で FFT 表示の安定化")),

        heading_2("2. 探索範囲 (search_band) の決定 — 5戦略比較"),
        paragraph(_text(
            "側面像では argmax が脊椎・心臓のエッジに locking する問題が発生したため、DIR-Lab 5ケースで比較:")),
        code("plain text",
             "戦略              成功率   平均NCC\n"
             "shroud_default    4/5      0.797\n"
             "shroud_narrow     5/5      0.990  ★ デフォルト採用\n"
             "shroud_lower      5/5      0.988\n"
             "intensity_roi     5/5      0.959\n"
             "frame_diff        3/5      0.653"),
        paragraph(_text("→ "), _text("search_band = (0.5, 0.95)", code=True),
                  _text(" を新デフォルトに決定。最も安定で全 5 ケース成功。")),

        heading_2("3. ペアサイクル抽出"),
        bullet(_text("各信号の end-of-expiration (local minimum) を境界として検出")),
        bullet(_text("各候補サイクルを共通長 64 にリサンプル → z-score → NCC 計算")),
        bullet(_text("最大 NCC の AP / Lateral ペアを採用")),
        bullet(_text("片方の信号が退化した場合、もう片方の境界を流用する fallback も実装")),

        heading_2("4. 動態シーケンス合成 (4DCT → 動態DRR)"),
        bullet(_text("DiffDRR で 各位相 (0%, 10%, ..., 90%) の AP / 側面 DRR をレンダリング")),
        bullet(_text("循環式位相カーブで N 呼吸サイクル分の時系列に展開")),
        bullet(_text("ジッタ・輝度ノイズで実画像感を付加")),
        bullet(_text("DRR の上下反転を補正 (頭側上)")),

        heading_1("公開データセットでの検証"),
        paragraph(
            _text("使用: "),
            _text("TCIA 4D-Lung", bold=True), _text(" (CC BY 3.0) 1症例 10位相、"),
            _text("DIR-Lab", bold=True), _text(" 10症例 (要登録、Dropbox 経由)。"),
        ),
        paragraph(_text(
            "raw データは git 管理外。派生レポート（DRR可視化・シュラウド・サイクルペア・メトリクス）のみ公開対象。")),

        heading_2("DIR-Lab 10ケース結果"),
        code("plain text",
             "Case   AP Hz   Lat Hz   NCC      AP cycle    Lat cycle\n"
             "case1  0.124   0.496    0.963    56-83       75-109\n"
             "case2  0.124   0.496    0.989    47-75       74-107\n"
             "case3  0.124   0.496    0.999    46-76       12-43\n"
             "case4  0.124   0.124    0.995    43-76       16-47\n"
             "case5  0.496   0.496    1.000    46-76       48-78\n"
             "case6  0.124   0.124    0.995    10-42       15-46\n"
             "case7  0.496   0.496    0.994    43-74       73-103\n"
             "case8  0.248   0.124    0.973    53-84       60-87\n"
             "case9  0.496   0.496    0.998    20-51       45-76\n"
             "case10 0.496   0.496    0.999    11-42       72-104\n"
             "\n"
             "mean NCC: 0.990   min: 0.963   max: 1.000\n"
             "success @ NCC>=0.9: 10/10"),
        paragraph(_text(
            "AP 周波数が 0.124 Hz と表示されるケースがあるが、これは 121 フレーム/15 fps の "
            "FFT bin 解像度 (≈0.124 Hz 刻み) に起因する表示上の問題。サイクル境界検出 (valley) "
            "は正しく動き、NCC ≥ 0.96 が品質を担保。")),

        heading_1("成果物"),
        bullet(_text("ライブレポート (GitHub Pages): "),
               _text(PAGES_URL, link=PAGES_URL)),
        bullet(_text("リポジトリ: "), _text(REPO_URL, link=REPO_URL)),
        bullet(_text("各ケース別の PDF + HTML レポート (4DCT位相プレビュー、位相別DRR、動的サンプル、"
                     "シュラウド × 2view、NCC マトリクス、選択ペア、メトリクス表)")),
        bullet(_text("Notion DB に主要メトリクスを自動投入 (このページのDB)")),
        bullet(_text("paired cycle 抽出済み .npy 配列 — そのまま 2D-4D モデルの入力に使用可能")),

        heading_1("自動化基盤"),
        paragraph(_text("git push 時に GitHub Actions が以下を実行:")),
        bullet(_text("outputs/* を _site/ に集約 (scripts/build_pages.py)")),
        bullet(_text("GitHub Pages へデプロイ")),
        bullet(_text("Notion API で DB に行追加 (scripts/notion_publish.py)")),
        bullet(_text("Slack webhook で通知 (未設定なら skip)")),

        heading_1("残課題と次のステップ"),
        bullet(_text("Slack webhook 設定 (incoming webhook URL を Secrets に追加するだけ)")),
        bullet(_text("実患者データでの最終検証")),
        bullet(_text("AP dominant_frequency の表示を FFT ではなくサイクル境界ベースに切替 (任意)")),
        bullet(_text("側面 search_band のケース別最適化 (case4 がやや低周波 lock)")),
        bullet(_text("Notion integration トークンのローテーション (チャット履歴に残ったため安全のため)")),

        divider(),
        callout("ℹ️",
                "本ページは scripts/notion_publish_summary.py で Notion API 経由で自動生成。"
                "次回マイルストーン時に同スクリプトの build_summary_* 関数を増やせば同様に投稿できる。"),
    ]
    return title, blocks


def main() -> int:
    token = os.environ.get("NOTION_API_KEY")
    db_id = os.environ.get("NOTION_DATABASE_ID")
    if not token or not db_id:
        print("Set NOTION_API_KEY and NOTION_DATABASE_ID env vars", file=sys.stderr)
        return 2

    title, blocks = build_summary_2026_05_20()
    body = {
        "parent": {"database_id": db_id},
        "properties": {
            "Title": {"title": [{"text": {"content": title}}]},
            "Source": {"select": {"name": "other"}},
            "Date": {"date": {"start": date.today().isoformat()}},
            "Run ID": {"rich_text": [{"text": {"content": f"summary_{date.today().isoformat()}"}}]},
            "Pages URL": {"url": PAGES_URL},
        },
        "children": blocks,
    }
    res = post("/pages", body, token)
    print(f"Created Notion page: {res.get('id')}")
    if res.get("url"):
        print(f"URL: {res.get('url')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
