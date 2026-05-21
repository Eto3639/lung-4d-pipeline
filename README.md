# Synthetic CT QA System

## 概要 (Overview)
本システムは、肺癌IGRT/ART（画像誘導/適応放射線治療）において、2Dレントゲン/透視画像から生成された**合成CT（Synthetic CT）**の品質と信頼性を評価するための包括的なQA（品質保証）ツールです。

標準的な評価指標（Gamma解析、SSIM、Jacobian Determinant等）を用い、臨床仕様の自動レポートを生成します。

## 機能 (Features)

システムは以下の4つの主要モジュールで構成されています：

1.  **幾何学的・構造的整合性 (Geometric Integrity)**
    *   **DRR再投影テスト**: 合成CTから仮想レントゲン（DRR）を作成し、入力画像とSSIM/NCCで比較。
    *   **断面連続性解析**: Z軸方向の不自然な段差を検知。
    *   **解剖学的指標**: 肺・骨の重心位置ズレを評価。

2.  **物理的・線量精度 (Dosimetric Accuracy)**
    *   **Gamma Analysis**: 線量分布の比較 (3mm/3%, 2mm/2%)。
    *   **DVHパラメータ**: $D_{95}$, $V_{20}$ 等の臨床指標比較とDVHプロット生成。
    *   **HU統計**: 組織別HU値の検証。

3.  **4D動態・時間的連続性 (Temporal Motion)**
    *   **Jacobian Determinant**: DIR（非硬性レジストレーション）を用いた局所的な体積保存性評価（Folding検知）。
    *   **軌跡解析**: 呼吸性移動の滑らかさ（Smoothness）を評価。

4.  **堅牢性・安全性 (Robustness Check)**
    *   **FOV/Masking**: 入力画像の異常（遮蔽、欠損）を検知。
    *   **SNR**: 画質のS/N比評価。

## 動作環境 (Requirements)

*   Python 3.10+
*   Linux (Ubuntu/CentOS recommended)
*   Dependencies:
    *   numpy, scipy, pandas
    *   scikit-image, SimpleITK
    *   reportlab (PDF生成)
    *   matplotlib (グラフ描画)
    *   pymedphys (Optional: 高速Gamma解析用)

## インストール (Installation)

```bash
# リポジトリのクローン
git clone <repository-url>
cd qa-system-sct

# 依存ライブラリのインストール
pip install -r requirements.txt
```

*(Note: `requirements.txt` は以下のパッケージを含みます: `numpy scipy pandas scikit-image SimpleITK reportlab matplotlib pymedphys`)*

## 使用方法 (Usage)

### 1. メインスクリプトの実行

サンプルのダミーデータを使用してQAパイプラインを実行し、レポートを生成します。

```bash
python3 src/main.py
```

### 2. ライブラリとしての利用

```python
from qa_system.dosimetric_accuracy import DosimetricAccuracy

# データ準備
data = {
    'synthetic_dose': dose_array,
    'reference_dose': ref_dose_array,
    'voxel_size': (2.0, 2.0, 2.0)
}

# モジュール実行
qa = DosimetricAccuracy()
results = qa.validate(data)
print(results)
```

### 3. 設定とキャリブレーション (Configuration & Calibration)

判定閾値は `config/thresholds.json` で管理されています。

#### 手動設定

`config/thresholds.json` を直接編集して、各モジュールのPass/Fail基準を変更できます。

```json
{
    "DosimetricAccuracy": {
        "gamma_3mm_3%_pass_rate_min": 0.95
    }
}
```

#### 自動キャリブレーション

基準データセット（Batch Data）を用いて統計的に適切な閾値（Mean ± 2SD）を算出するツールが含まれています。

```bash
# キャリブレーションツールの実行
python3 src/tools/calibrate.py
```

実行後、`config/suggested_thresholds.json` が生成されます。内容を確認し、`thresholds.json` に適用してください。

## 出力 (Outputs)

*   **PDFレポート**: `reports/QA_Report_{PatientID}.pdf`
    *   各モジュールのPass/Fail判定
    *   詳細数値メトリクス
    *   DVH比較グラフ
    *   DRRオーバーレイ画像

## アーキテクチャ

*   `src/qa_system/`: QAモジュールのソースコード
*   `src/main.py`: オーケストレーション（並列処理）
*   `tests/`: ユニットテスト

詳細は `DESIGN.md` を参照してください。
