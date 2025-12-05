# 🎙️ Llasa-TTS-8B WebUI Demo

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)](https://developer.nvidia.com/cuda-downloads)

> High-quality Text-to-Speech system based on Llasa-8B with intelligent GPU memory management

## ✨ Features

- 🚀 **Intelligent GPU Management**: Lazy loading + instant offload, reducing idle GPU memory by 96% (from 24GB to <1GB)
- 🎨 **Three Access Modes**: Web UI (Gradio) + REST API (Flask) + MCP (Model Context Protocol)
- 🔄 **Auto GPU Selection**: Automatically selects the GPU with the least memory usage
- 🌍 **Multi-language Support**: Chinese, English, and mixed-language speech generation
- 🎭 **Voice Cloning**: High-quality voice cloning based on reference audio
- 🐳 **One-Click Deployment**: Docker + docker-compose for production-ready deployment
- ⚡ **Optimized Performance**: Faster-Whisper for ASR, 500% faster than official Whisper

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

### Prerequisites

- Linux system (Ubuntu 20.04+ recommended)
- NVIDIA GPU (24GB+ VRAM)
- Docker + Docker Compose + nvidia-docker

### One-Command Launch

```bash
git clone https://github.com/yourusername/llasa-tts-8b-webui.git
cd llasa-tts-8b-webui
chmod +x start.sh
./start.sh
```

Access the services:
- **Web UI**: http://localhost:7860
- **API**: http://localhost:7861
- **API Docs**: http://localhost:7861/apidocs

## 📦 Installation

### Method 1: Docker Deployment (Recommended)

**Step 1: Clone the repository**
```bash
git clone https://github.com/yourusername/llasa-tts-8b-webui.git
cd llasa-tts-8b-webui
```

**Step 2: Configure environment**
```bash
cp .env.example .env
# Edit .env to set your configuration
```

**Step 3: Start services**
```bash
./start.sh
```

The script will:
- ✅ Check Docker environment
- ✅ Auto-select the least busy GPU
- ✅ Build Docker image
- ✅ Start containers
- ✅ Display access information

**Docker Compose Example:**
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

**Docker Run Command:**
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

### Method 2: Conda Environment

**Step 1: Create environment**
```bash
conda create -n llasa-tts python=3.9
conda activate llasa-tts
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Run application**
```bash
python main.py
```

Access: http://localhost:7860

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_IDLE_TIMEOUT` | 600 | GPU idle timeout (seconds) |
| `UI_PORT` | 7860 | Web UI port |
| `API_PORT` | 7861 | REST API port |
| `HF_TOKEN` | - | HuggingFace token (optional) |
| `HF_ENDPOINT` | https://hf-mirror.com | HuggingFace mirror |
| `LLASA_MODEL_PATH` | HKUSTAudio/Llasa-8B | Llasa model path |
| `XCODEC_MODEL_PATH` | HKUSTAudio/xcodec2 | XCodec2 model path |
| `WHISPER_MODEL_PATH` | Systran/faster-whisper-large-v3 | Whisper model path |

### Configuration File

Create `.env` from template:
```bash
cp .env.example .env
```

Edit `.env`:
```bash
# GPU Configuration
NVIDIA_VISIBLE_DEVICES=0
GPU_IDLE_TIMEOUT=600

# Port Configuration
UI_PORT=7860
API_PORT=7861

# HuggingFace Configuration
HF_ENDPOINT=https://hf-mirror.com
# HF_TOKEN=your_token_here

# Model Paths (optional, use local models)
# LLASA_MODEL_PATH=/path/to/Llasa-8B
# XCODEC_MODEL_PATH=/path/to/xcodec2
```

## 📖 Usage

### Web UI

1. Open http://localhost:7860
2. Upload reference audio (15-20 seconds, WAV format)
3. Click "Auto Transcribe" or manually input reference text
4. Enter target text to generate
5. Click "Generate Speech"

### REST API

**Health Check:**
```bash
curl http://localhost:7861/health
```

**Generate Speech:**
```bash
curl -X POST http://localhost:7861/api/tts \
  -F "audio=@reference.wav" \
  -F "ref_text=Reference audio text" \
  -F "target_text=Text to generate" \
  --output generated.wav
```

**GPU Status:**
```bash
curl http://localhost:7861/api/gpu/status
```

**Manual Offload:**
```bash
curl -X POST http://localhost:7861/api/gpu/offload
```

### MCP (Model Context Protocol)

Run MCP server:
```bash
docker exec -it llasa-tts-8b-webui python mcp_server.py
```

Or on host:
```bash
python mcp_server.py
```

Available tools:
- `generate_speech()` - Generate speech
- `transcribe_audio()` - Transcribe audio
- `get_gpu_status()` - Get GPU status
- `offload_gpu()` - Offload GPU memory
- `release_gpu()` - Release GPU completely

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/gpu/status` | Get GPU status |
| POST | `/api/gpu/offload` | Offload models to CPU |
| POST | `/api/gpu/release` | Release all models |
| POST | `/api/transcribe` | Transcribe audio (ASR) |
| POST | `/api/tts` | Generate speech (TTS) |
| GET | `/apidocs` | Swagger documentation |

### API Examples

See [API Documentation](http://localhost:7861/apidocs) for interactive examples.

## 🛠️ Tech Stack

### Core Technologies
- **PyTorch 2.6.0** - Deep learning framework
- **Transformers 4.45.2** - Model loading
- **Gradio 4.0+** - Web UI
- **Flask 3.0.0** - REST API
- **FastMCP** - MCP server

### Models
- **Llasa-8B** - Speech generation (~17GB)
- **XCodec2** - Audio codec (~3GB)
- **Faster-Whisper** - Speech recognition (~3GB, CPU)

### Deployment
- **Docker** - Containerization
- **Docker Compose** - Orchestration
- **NVIDIA Docker** - GPU support

## 📊 Performance

### GPU Memory Usage

| Stage | Traditional | Smart Management | Savings |
|-------|------------|------------------|---------|
| Startup | 24 GB | 0 GB | 100% |
| Running | 24 GB | 24 GB | 0% |
| Idle | 24 GB | < 1 GB | **96%** |

### Loading Times (RTX 4090)

- First load: 20-30 seconds
- CPU → GPU: 2-5 seconds
- GPU → CPU: 2 seconds
- Complete release: 1 second

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Changelog

### v1.0.0 (2025-12-06)
- ✨ Initial release
- 🚀 Intelligent GPU memory management
- 🎨 Three access modes (UI + API + MCP)
- 🔄 Auto GPU selection
- 🐳 Docker deployment

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: Models and dependencies have their own licenses:
- Llasa-8B: [Official License](https://huggingface.co/HKUSTAudio/Llasa-8B)
- XCodec2: [Official License](https://huggingface.co/HKUSTAudio/xcodec2)
- Faster-Whisper: Apache 2.0

## 🙏 Acknowledgments

- Original project: [HKUSTAudio/Llasa-8B](https://huggingface.co/HKUSTAudio/Llasa-8B)
- Thanks to all contributors and the open-source community

## 📞 Contact & Support

- 📧 Issues: [GitHub Issues](https://github.com/yourusername/llasa-tts-8b-webui/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/llasa-tts-8b-webui/discussions)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/llasa-tts-8b-webui&type=Date)](https://star-history.com/#yourusername/llasa-tts-8b-webui)

## 📱 Follow Us

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

---

<p align="center">Made with ❤️ by the Llasa-TTS-8B Team</p>
