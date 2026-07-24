# CNN Visualize

CIFAR-10 データセットを用いた CNN の学習・可視化サンプルスクリプト。  
学習プロセスの各段階でネットワーク内部の状態を PNG ファイルとして出力し、CNN の動作を視覚的に理解することを目的とする。

## セットアップ

```bash
pip install -r requiements.txt
python cnn_visualize.py
```

CIFAR-10 データセットは初回実行時に `data/` へ自動ダウンロードされる。

---

## ファイル構成

```
cnn-ws/
├── cnn_visualize.py          # メインスクリプト
├── requiements.txt           # 依存ライブラリ
├── data/
│   └── cifar-10-batches-py/  # CIFAR-10 データセット（自動ダウンロード）
└── output/                   # 生成された可視化画像（7枚）
    ├── 01_network_architecture.png
    ├── 02_conv1_filters.png
    ├── 03_training_curve.png
    ├── 04_conv1_feature_maps.png
    ├── 05_conv2_feature_maps.png
    ├── 06_activation_comparison.png
    └── 07_test_predictions.png
```

---

## ネットワーク構造（SimpleCNN）

| レイヤー | 種類 | 設定 | 出力サイズ |
|---------|------|------|-----------|
| 入力 | — | — | 32×32×3 |
| Conv1 | Conv2d | in=3, out=16, kernel=3×3, padding=1 | 32×32×16 |
| ReLU | 活性化関数 | — | 32×32×16 |
| Pool1 | MaxPool2d | kernel=2×2, stride=2 | 16×16×16 |
| Conv2 | Conv2d | in=16, out=32, kernel=3×3, padding=1 | 16×16×32 |
| ReLU | 活性化関数 | — | 16×16×32 |
| Pool2 | MaxPool2d | kernel=2×2, stride=2 | 8×8×32 |
| Flatten | — | — | 2048 |
| FC1 | Linear | in=2048, out=256 | 256 |
| ReLU | 活性化関数 | — | 256 |
| FC2 | Linear | in=256, out=10 | 10（クラス数） |

### 特殊メソッド：`get_feature_maps(x)`

`forward()` に加え、中間層の出力を辞書形式で返す補助メソッド。可視化用途専用。

| キー | 内容 |
|-----|------|
| `conv1_before_relu` | Conv1 適用後・ReLU 前 |
| `conv1_after_relu` | ReLU 後 |
| `pool1` | MaxPool1 後 |
| `conv2_before_relu` | Conv2 適用後・ReLU 前 |
| `conv2_after_relu` | ReLU 後 |
| `pool2` | MaxPool2 後 |

---

## データセット

| 項目 | 値 |
|-----|---|
| データセット | CIFAR-10 |
| 自動ダウンロード | あり（`data/` フォルダへ） |
| 訓練データ数 | 50,000 枚 |
| テストデータ数 | 10,000 枚 |
| クラス数 | 10（airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck） |

### 前処理

| フェーズ | 処理 |
|---------|------|
| 訓練 | RandomHorizontalFlip → RandomCrop(32, padding=4) → ToTensor → Normalize |
| テスト | ToTensor → Normalize |

正規化パラメータ（CIFAR-10 統計値）：
- mean: `(0.4914, 0.4822, 0.4465)`
- std: `(0.2470, 0.2435, 0.2616)`

---

## ハイパーパラメータ

| パラメータ | 値 |
|-----------|---|
| エポック数 | 10 |
| バッチサイズ | 64 |
| 学習率 | 0.001 |
| オプティマイザ | Adam |
| 損失関数 | CrossEntropyLoss |
| デバイス | CUDA（利用可能な場合）/ CPU |

---

## 実行フロー

スクリプトは以下の 7 ステップを順番に実行する。

```
[1/7] ネットワーク構造図を保存（学習前）
[2/7] CIFAR-10 データセットを準備
[3/7] Conv1 フィルタを保存（学習前の初期状態）
[4/7] 学習（10エポック）
[5/7] 学習曲線を保存
[6/7] 特徴マップ・活性化分布を保存（テスト画像 1枚）
[7/7] テスト画像の予測結果を保存
```

---

## 出力ファイル詳細

### `01_network_architecture.png` — ネットワーク構造図

- 関数: `save_network_architecture()`
- 内容: 各層をカラーブロックで表現した横並びアーキテクチャ図
- 生成タイミング: **学習前**

### `02_conv1_filters.png` — Conv1 フィルタ重み

- 関数: `save_conv1_filters(model)`
- 内容: Conv1 の 16 個のフィルタ（3×3×RGB）を 2行×8列で表示
- 各フィルタは `[0, 1]` に正規化して RGB 画像として表示
- 生成タイミング: **学習前**（初期ランダム重み）

### `03_training_curve.png` — 学習曲線

- 関数: `save_training_curve(history)`
- 内容: 左グラフ（Loss）・右グラフ（Accuracy）の 2 パネル
  - 各グラフに Train / Val の 2 系列をプロット
- 生成タイミング: **学習後**

### `04_conv1_feature_maps.png` — Conv1 特徴マップ

- 関数: `save_feature_maps(model, image_tensor, label_idx, layer="conv1")`
- 内容: テスト画像 1枚（`testset[0]`）の Conv1 出力（ReLU 後）を 16 チャンネル分表示
- 入力画像も左上に表示（逆正規化して表示）
- カラーマップ: viridis（明るいほど強い活性化）
- 生成タイミング: **学習後**

### `05_conv2_feature_maps.png` — Conv2 特徴マップ

- 関数: `save_feature_maps(model, image_tensor, label_idx, layer="conv2")`
- 内容: 同一テスト画像の Conv2 出力（ReLU 後）を 32 チャンネル分表示
- 生成タイミング: **学習後**

### `06_activation_comparison.png` — ReLU 前後の活性化分布

- 関数: `save_activation_comparison(model, image_tensor)`
- 内容: 2行×2列のヒストグラム
  - Conv1 / Conv2 それぞれで ReLU 前後を比較
  - ReLU 前: 負の値を含む分布（青色）、x=0 に赤い垂直線
  - ReLU 後: 負の値がゼロにクリップされた分布（緑色）、ゼロ率を表示
- 生成タイミング: **学習後**

### `07_test_predictions.png` — テスト画像の予測結果

- 関数: `save_test_predictions(model, testset, n=16)`
- 内容: テストセット先頭 16 枚を 2行×8列で表示
  - 正解: 緑の枠線 / 不正解: 赤の枠線
  - タイトルに正解ラベル（GT）と予測ラベル（Pred）を表示
- 生成タイミング: **学習後**

---

## 依存ライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| torch | >=2.0.0 | モデル定義・学習 |
| torchvision | >=0.15.0 | CIFAR-10 データセット・前処理 |
| matplotlib | >=3.7.0 | 可視化・PNG 保存 |
| numpy | >=1.24.0 | 配列操作 |
| Pillow | >=9.0.0 | 画像処理（torchvision 依存） |

matplotlib は非インタラクティブバックエンド（`Agg`）を使用するため、GUI 環境不要。

---

## 補足

- 出力ディレクトリ (`output/`) は起動時に自動作成される。
- CIFAR-10 データは初回実行時に自動ダウンロードされ `data/` に保存される（`data/` は `.gitignore` 除外済み）。
- `save_test_predictions` はテストセット先頭 16 枚を固定で使用するため、ブランチ間でモデルを変えても同じ画像で比較できる。
- matplotlib は非インタラクティブバックエンド（`Agg`）を使用するため、GUI 環境不要。
