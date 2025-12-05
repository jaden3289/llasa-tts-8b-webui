# 🎙️ Llasa-TTS-8B WebUI 演示

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)](https://developer.nvidia.com/cuda-downloads)

> 基於 Llasa-8B 的高品質文字轉語音系統，具有智慧 GPU 顯存管理功能

## ✨ 功能特性

- 🚀 **智慧 GPU 管理**：懶載入 + 即用即卸，閒置時顯存佔用降低 96%（從 24GB 降至 <1GB）
- 🎨 **三種存取模式**：Web UI（Gradio）+ REST API（Flask）+ MCP（模型上下文協定）
- 🔄 **自動 GPU 選擇**：自動選擇顯存佔用最少的 GPU
- 🌍 **多語言支援**：支援中文、英文及混合語音生成
- 🎭 **語音複製**：基於參考音訊的高品質語音複製
- 🐳 **一鍵部署**：Docker + docker-compose 生產級部署
- ⚡ **效能最佳化**：使用 Faster-Whisper 進行 ASR，比官方 Whisper 快 500%

## 📋 目錄

- [快速開始](#-快速開始)
- [安裝部署](#-安裝部署)
- [設定說明](#-設定說明)
- [使用方法](#-使用方法)
- [API 文件](#-api-文件)
- [技術棧](#-技術棧)
- [貢獻指南](#-貢獻指南)
- [授權條款](#-授權條款)

## 🚀 快速開始

### 前置要求

- Linux 系統（推薦 Ubuntu 20.04+）
- NVIDIA GPU（24GB+ 顯存）
- Docker + Docker Compose + nvidia-docker

### 一鍵啟動

```bash
git clone https://github.com/yourusername/llasa-tts-8b-webui.git
cd llasa-tts-8b-webui
chmod +x start.sh
./start.sh
```

存取服務：
- **Web UI**：http://localhost:7860
- **API**：http://localhost:7861
- **API 文件**：http://localhost:7861/apidocs

## 📦 安裝部署

### 方式一：Docker 部署（推薦）

**步驟 1：複製儲存庫**
```bash
git clone https://github.com/yourusername/llasa-tts-8b-webui.git
cd llasa-tts-8b-webui
```

**步驟 2：設定環境**
```bash
cp .env.example .env
# 編輯 .env 設定您的配置
```

**步驟 3：啟動服務**
```bash
./start.sh
```

腳本將自動：
- ✅ 檢查 Docker 環境
- ✅ 自動選擇最空閒的 GPU
- ✅ 建置 Docker 映像
- ✅ 啟動容器
- ✅ 顯示存取資訊

**Docker Compose 範例：**
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

**Docker Run 指令：**
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

### 方式二：Conda 環境

**步驟 1：建立環境**
```bash
conda create -n llasa-tts python=3.9
conda activate llasa-tts
```

**步驟 2：安裝相依套件**
```bash
pip install -r requirements.txt
```

**步驟 3：執行應用程式**
```bash
python main.py
```

存取：http://localhost:7860

## ⚙️ 設定說明

### 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `GPU_IDLE_TIMEOUT` | 600 | GPU 閒置逾時（秒） |
| `UI_PORT` | 7860 | Web UI 連接埠 |
| `API_PORT` | 7861 | REST API 連接埠 |
| `HF_TOKEN` | - | HuggingFace token（選用） |
| `HF_ENDPOINT` | https://hf-mirror.com | HuggingFace 映像 |
| `LLASA_MODEL_PATH` | HKUSTAudio/Llasa-8B | Llasa 模型路徑 |
| `XCODEC_MODEL_PATH` | HKUSTAudio/xcodec2 | XCodec2 模型路徑 |
| `WHISPER_MODEL_PATH` | Systran/faster-whisper-large-v3 | Whisper 模型路徑 |

### 設定檔

從範本建立 `.env`：
```bash
cp .env.example .env
```

編輯 `.env`：
```bash
# GPU 設定
NVIDIA_VISIBLE_DEVICES=0
GPU_IDLE_TIMEOUT=600

# 連接埠設定
UI_PORT=7860
API_PORT=7861

# HuggingFace 設定
HF_ENDPOINT=https://hf-mirror.com
# HF_TOKEN=your_token_here

# 模型路徑（選用，使用本機模型）
# LLASA_MODEL_PATH=/path/to/Llasa-8B
# XCODEC_MODEL_PATH=/path/to/xcodec2
```

## 📖 使用方法

### Web UI

1. 開啟 http://localhost:7860
2. 上傳參考音訊（15-20 秒，WAV 格式）
3. 點選「自動轉錄」或手動輸入參考文字
4. 輸入要生成的目標文字
5. 點選「生成語音」

### REST API

**健康檢查：**
```bash
curl http://localhost:7861/health
```

**生成語音：**
```bash
curl -X POST http://localhost:7861/api/tts \
  -F "audio=@reference.wav" \
  -F "ref_text=參考音訊文字" \
  -F "target_text=要生成的文字" \
  --output generated.wav
```

**GPU 狀態：**
```bash
curl http://localhost:7861/api/gpu/status
```

**手動卸載：**
```bash
curl -X POST http://localhost:7861/api/gpu/offload
```

### MCP（模型上下文協定）

執行 MCP 伺服器：
```bash
docker exec -it llasa-tts-8b-webui python mcp_server.py
```

或在主機執行：
```bash
python mcp_server.py
```

可用工具：
- `generate_speech()` - 生成語音
- `transcribe_audio()` - 轉錄音訊
- `get_gpu_status()` - 取得 GPU 狀態
- `offload_gpu()` - 卸載 GPU 顯存
- `release_gpu()` - 完全釋放 GPU

## 📚 API 文件

### 端點

| 方法 | 端點 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| GET | `/api/gpu/status` | 取得 GPU 狀態 |
| POST | `/api/gpu/offload` | 卸載模型到 CPU |
| POST | `/api/gpu/release` | 釋放所有模型 |
| POST | `/api/transcribe` | 轉錄音訊（ASR） |
| POST | `/api/tts` | 生成語音（TTS） |
| GET | `/apidocs` | Swagger 文件 |

### API 範例

檢視 [API 文件](http://localhost:7861/apidocs) 取得互動式範例。

## 🛠️ 技術棧

### 核心技術
- **PyTorch 2.6.0** - 深度學習框架
- **Transformers 4.45.2** - 模型載入
- **Gradio 4.0+** - Web UI
- **Flask 3.0.0** - REST API
- **FastMCP** - MCP 伺服器

### 模型
- **Llasa-8B** - 語音生成（~17GB）
- **XCodec2** - 音訊編解碼（~3GB）
- **Faster-Whisper** - 語音辨識（~3GB，CPU）

### 部署
- **Docker** - 容器化
- **Docker Compose** - 編排
- **NVIDIA Docker** - GPU 支援

## 📊 效能指標

### GPU 顯存使用

| 階段 | 傳統方式 | 智慧管理 | 節省 |
|------|---------|---------|------|
| 啟動 | 24 GB | 0 GB | 100% |
| 執行中 | 24 GB | 24 GB | 0% |
| 閒置時 | 24 GB | < 1 GB | **96%** |

### 載入時間（RTX 4090）

- 首次載入：20-30 秒
- CPU → GPU：2-5 秒
- GPU → CPU：2 秒
- 完全釋放：1 秒

## 🤝 貢獻指南

歡迎貢獻！請遵循以下步驟：

1. Fork 本儲存庫
2. 建立您的特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交您的變更（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 開啟一個 Pull Request

## 📝 更新日誌

### v1.0.0 (2025-12-06)
- ✨ 初始版本發布
- 🚀 智慧 GPU 顯存管理
- 🎨 三種存取模式（UI + API + MCP）
- 🔄 自動 GPU 選擇
- 🐳 Docker 部署

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案。

**注意**：模型和相依套件有各自的授權條款：
- Llasa-8B：[官方授權條款](https://huggingface.co/HKUSTAudio/Llasa-8B)
- XCodec2：[官方授權條款](https://huggingface.co/HKUSTAudio/xcodec2)
- Faster-Whisper：Apache 2.0

## 🙏 致謝

- 原始專案：[HKUSTAudio/Llasa-8B](https://huggingface.co/HKUSTAudio/Llasa-8B)
- 感謝所有貢獻者和開源社群

## 📞 聯絡與支援

- 📧 問題回饋：[GitHub Issues](https://github.com/yourusername/llasa-tts-8b-webui/issues)
- 💬 討論區：[GitHub Discussions](https://github.com/yourusername/llasa-tts-8b-webui/discussions)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/llasa-tts-8b-webui&type=Date)](https://star-history.com/#yourusername/llasa-tts-8b-webui)

## 📱 關注公眾號

![公眾號](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

---

<p align="center">用 ❤️ 製作 by Llasa-TTS-8B 團隊</p>
