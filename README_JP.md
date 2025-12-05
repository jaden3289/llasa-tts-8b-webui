# 🎙️ Llasa-TTS-8B WebUI デモ

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)](https://developer.nvidia.com/cuda-downloads)

> Llasa-8B ベースの高品質テキスト読み上げシステム、インテリジェント GPU メモリ管理機能付き

## ✨ 機能

- 🚀 **インテリジェント GPU 管理**：遅延ロード + 即時アンロード、アイドル時の GPU メモリ使用量を 96% 削減（24GB から <1GB へ）
- 🎨 **3つのアクセスモード**：Web UI（Gradio）+ REST API（Flask）+ MCP（Model Context Protocol）
- 🔄 **自動 GPU 選択**：メモリ使用量が最も少ない GPU を自動選択
- 🌍 **多言語対応**：中国語、英語、混合言語の音声生成をサポート
- 🎭 **音声クローニング**：参照音声に基づく高品質な音声クローニング
- 🐳 **ワンクリックデプロイ**：Docker + docker-compose による本番環境対応デプロイ
- ⚡ **パフォーマンス最適化**：ASR に Faster-Whisper を使用、公式 Whisper より 500% 高速

## 📋 目次

- [クイックスタート](#-クイックスタート)
- [インストール](#-インストール)
- [設定](#-設定)
- [使用方法](#-使用方法)
- [API ドキュメント](#-api-ドキュメント)
- [技術スタック](#-技術スタック)
- [貢献](#-貢献)
- [ライセンス](#-ライセンス)

## 🚀 クイックスタート

### 前提条件

- Linux システム（Ubuntu 20.04+ 推奨）
- NVIDIA GPU（24GB+ VRAM）
- Docker + Docker Compose + nvidia-docker

### ワンコマンド起動

```bash
git clone https://github.com/yourusername/llasa-tts-8b-webui.git
cd llasa-tts-8b-webui
chmod +x start.sh
./start.sh
```

サービスへのアクセス：
- **Web UI**：http://localhost:7860
- **API**：http://localhost:7861
- **API ドキュメント**：http://localhost:7861/apidocs

## 📦 インストール

### 方法 1：Docker デプロイ（推奨）

**ステップ 1：リポジトリのクローン**
```bash
git clone https://github.com/yourusername/llasa-tts-8b-webui.git
cd llasa-tts-8b-webui
```

**ステップ 2：環境設定**
```bash
cp .env.example .env
# .env を編集して設定を行う
```

**ステップ 3：サービス起動**
```bash
./start.sh
```

スクリプトは自動的に：
- ✅ Docker 環境をチェック
- ✅ 最も空いている GPU を自動選択
- ✅ Docker イメージをビルド
- ✅ コンテナを起動
- ✅ アクセス情報を表示

**Docker Compose の例：**
```yaml
version: '3.8'
services:
  llasa-tts-webui:
    image: llasa-tts-8b:latest
    container_name: llasa-tts-8b-webui
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    ports:
      - "7860:7860"  # Web UI
      - "7861:7861"  # REST API
    volumes:
      - ./models_cache:/root/.cache/huggingface
      - ./outputs:/app/outputs
    environment:
      - GPU_IDLE_TIMEOUT=600
      - HF_ENDPOINT=https://hf-mirror.com
    restart: unless-stopped
```

**Docker Run コマンド：**
```bash
docker run -d \
  --name llasa-tts-8b \
  --gpus '"device=0"' \
  -p 7860:7860 \
  -p 7861:7861 \
  -v $(pwd)/models_cache:/root/.cache/huggingface \
  -v $(pwd)/outputs:/app/outputs \
  -e GPU_IDLE_TIMEOUT=600 \
  llasa-tts-8b:latest
```

### 方法 2：Conda 環境

**ステップ 1：環境作成**
```bash
conda create -n llasa-tts python=3.9
conda activate llasa-tts
```

**ステップ 2：依存関係のインストール**
```bash
pip install -r requirements.txt
```

**ステップ 3：アプリケーション実行**
```bash
python main.py
```

アクセス：http://localhost:7860

## ⚙️ 設定

### 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `GPU_IDLE_TIMEOUT` | 600 | GPU アイドルタイムアウト（秒） |
| `UI_PORT` | 7860 | Web UI ポート |
| `API_PORT` | 7861 | REST API ポート |
| `HF_TOKEN` | - | HuggingFace トークン（オプション） |
| `HF_ENDPOINT` | https://hf-mirror.com | HuggingFace ミラー |
| `LLASA_MODEL_PATH` | HKUSTAudio/Llasa-8B | Llasa モデルパス |
| `XCODEC_MODEL_PATH` | HKUSTAudio/xcodec2 | XCodec2 モデルパス |
| `WHISPER_MODEL_PATH` | Systran/faster-whisper-large-v3 | Whisper モデルパス |

### 設定ファイル

テンプレートから `.env` を作成：
```bash
cp .env.example .env
```

`.env` を編集：
```bash
# GPU 設定
NVIDIA_VISIBLE_DEVICES=0
GPU_IDLE_TIMEOUT=600

# ポート設定
UI_PORT=7860
API_PORT=7861

# HuggingFace 設定
HF_ENDPOINT=https://hf-mirror.com
# HF_TOKEN=your_token_here

# モデルパス（オプション、ローカルモデルを使用）
# LLASA_MODEL_PATH=/path/to/Llasa-8B
# XCODEC_MODEL_PATH=/path/to/xcodec2
```

## 📖 使用方法

### Web UI

1. http://localhost:7860 を開く
2. 参照音声をアップロード（15-20秒、WAV形式）
3. 「自動文字起こし」をクリックするか、手動で参照テキストを入力
4. 生成するターゲットテキストを入力
5. 「音声生成」をクリック

### REST API

**ヘルスチェック：**
```bash
curl http://localhost:7861/health
```

**音声生成：**
```bash
curl -X POST http://localhost:7861/api/tts \
  -F "audio=@reference.wav" \
  -F "ref_text=参照音声のテキスト" \
  -F "target_text=生成するテキスト" \
  --output generated.wav
```

**GPU ステータス：**
```bash
curl http://localhost:7861/api/gpu/status
```

**手動アンロード：**
```bash
curl -X POST http://localhost:7861/api/gpu/offload
```

### MCP（Model Context Protocol）

MCP サーバーを実行：
```bash
docker exec -it llasa-tts-8b-webui python mcp_server.py
```

またはホストで実行：
```bash
python mcp_server.py
```

利用可能なツール：
- `generate_speech()` - 音声生成
- `transcribe_audio()` - 音声文字起こし
- `get_gpu_status()` - GPU ステータス取得
- `offload_gpu()` - GPU メモリアンロード
- `release_gpu()` - GPU 完全解放

## 📚 API ドキュメント

### エンドポイント

| メソッド | エンドポイント | 説明 |
|---------|--------------|------|
| GET | `/health` | ヘルスチェック |
| GET | `/api/gpu/status` | GPU ステータス取得 |
| POST | `/api/gpu/offload` | モデルを CPU にアンロード |
| POST | `/api/gpu/release` | すべてのモデルを解放 |
| POST | `/api/transcribe` | 音声文字起こし（ASR） |
| POST | `/api/tts` | 音声生成（TTS） |
| GET | `/apidocs` | Swagger ドキュメント |

### API 例

インタラクティブな例については [API ドキュメント](http://localhost:7861/apidocs) を参照してください。

## 🛠️ 技術スタック

### コア技術
- **PyTorch 2.6.0** - ディープラーニングフレームワーク
- **Transformers 4.45.2** - モデルローディング
- **Gradio 4.0+** - Web UI
- **Flask 3.0.0** - REST API
- **FastMCP** - MCP サーバー

### モデル
- **Llasa-8B** - 音声生成（~17GB）
- **XCodec2** - オーディオコーデック（~3GB）
- **Faster-Whisper** - 音声認識（~3GB、CPU）

### デプロイ
- **Docker** - コンテナ化
- **Docker Compose** - オーケストレーション
- **NVIDIA Docker** - GPU サポート

## 📊 パフォーマンス

### GPU メモリ使用量

| ステージ | 従来の方法 | スマート管理 | 削減率 |
|---------|-----------|-------------|--------|
| 起動時 | 24 GB | 0 GB | 100% |
| 実行中 | 24 GB | 24 GB | 0% |
| アイドル時 | 24 GB | < 1 GB | **96%** |

### ロード時間（RTX 4090）

- 初回ロード：20-30秒
- CPU → GPU：2-5秒
- GPU → CPU：2秒
- 完全解放：1秒

## 🤝 貢献

貢献を歓迎します！以下の手順に従ってください：

1. リポジトリをフォーク
2. フィーチャーブランチを作成（`git checkout -b feature/AmazingFeature`）
3. 変更をコミット（`git commit -m 'Add some AmazingFeature'`）
4. ブランチにプッシュ（`git push origin feature/AmazingFeature`）
5. プルリクエストを開く

## 📝 変更履歴

### v1.0.0 (2025-12-06)
- ✨ 初回リリース
- 🚀 インテリジェント GPU メモリ管理
- 🎨 3つのアクセスモード（UI + API + MCP）
- 🔄 自動 GPU 選択
- 🐳 Docker デプロイ

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下でライセンスされています - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

**注意**：モデルと依存関係には独自のライセンスがあります：
- Llasa-8B：[公式ライセンス](https://huggingface.co/HKUSTAudio/Llasa-8B)
- XCodec2：[公式ライセンス](https://huggingface.co/HKUSTAudio/xcodec2)
- Faster-Whisper：Apache 2.0

## 🙏 謝辞

- オリジナルプロジェクト：[HKUSTAudio/Llasa-8B](https://huggingface.co/HKUSTAudio/Llasa-8B)
- すべての貢献者とオープンソースコミュニティに感謝

## 📞 お問い合わせとサポート

- 📧 問題報告：[GitHub Issues](https://github.com/yourusername/llasa-tts-8b-webui/issues)
- 💬 ディスカッション：[GitHub Discussions](https://github.com/yourusername/llasa-tts-8b-webui/discussions)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/llasa-tts-8b-webui&type=Date)](https://star-history.com/#yourusername/llasa-tts-8b-webui)

## 📱 公式アカウントをフォロー

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

---

<p align="center">❤️ を込めて Llasa-TTS-8B チームが作成</p>
