# 🎙️ Llasa-TTS-8B WebUI 演示

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)](https://developer.nvidia.com/cuda-downloads)

> 基于 Llasa-8B 的高质量文本转语音系统，具有智能 GPU 显存管理功能

## ✨ 功能特性

- 🚀 **智能 GPU 管理**：懒加载 + 即用即卸，空闲时显存占用降低 96%（从 24GB 降至 <1GB）
- 🎨 **三种访问模式**：Web UI（Gradio）+ REST API（Flask）+ MCP（模型上下文协议）
- 🔄 **自动 GPU 选择**：自动选择显存占用最少的 GPU
- 🌍 **多语言支持**：支持中文、英文及混合语音生成
- 🎭 **语音克隆**：基于参考音频的高质量语音克隆
- 🐳 **一键部署**：Docker + docker-compose 生产级部署
- ⚡ **性能优化**：使用 Faster-Whisper 进行 ASR，比官方 Whisper 快 500%

## 📋 目录

- [快速开始](#-快速开始)
- [安装部署](#-安装部署)
- [配置说明](#-配置说明)
- [使用方法](#-使用方法)
- [API 文档](#-api-文档)
- [技术栈](#-技术栈)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

## 🚀 快速开始

### 前置要求

- Linux 系统（推荐 Ubuntu 20.04+）
- NVIDIA GPU（24GB+ 显存）
- Docker + Docker Compose + nvidia-docker

### 一键启动

```bash
git clone https://github.com/yourusername/llasa-tts-8b-webui.git
cd llasa-tts-8b-webui
chmod +x start.sh
./start.sh
```

访问服务：
- **Web UI**：http://localhost:7860
- **API**：http://localhost:7861
- **API 文档**：http://localhost:7861/apidocs

## 📦 安装部署

### 方式一：Docker 部署（推荐）

**步骤 1：克隆仓库**
```bash
git clone https://github.com/yourusername/llasa-tts-8b-webui.git
cd llasa-tts-8b-webui
```

**步骤 2：配置环境**
```bash
cp .env.example .env
# 编辑 .env 设置您的配置
```

**步骤 3：启动服务**
```bash
./start.sh
```

脚本将自动：
- ✅ 检查 Docker 环境
- ✅ 自动选择最空闲的 GPU
- ✅ 构建 Docker 镜像
- ✅ 启动容器
- ✅ 显示访问信息

**Docker Compose 示例：**
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

**Docker Run 命令：**
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

### 方式二：Conda 环境

**步骤 1：创建环境**
```bash
conda create -n llasa-tts python=3.9
conda activate llasa-tts
```

**步骤 2：安装依赖**
```bash
pip install -r requirements.txt
```

**步骤 3：运行应用**
```bash
python main.py
```

访问：http://localhost:7860

## ⚙️ 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPU_IDLE_TIMEOUT` | 600 | GPU 空闲超时（秒） |
| `UI_PORT` | 7860 | Web UI 端口 |
| `API_PORT` | 7861 | REST API 端口 |
| `HF_TOKEN` | - | HuggingFace token（可选） |
| `HF_ENDPOINT` | https://hf-mirror.com | HuggingFace 镜像 |
| `LLASA_MODEL_PATH` | HKUSTAudio/Llasa-8B | Llasa 模型路径 |
| `XCODEC_MODEL_PATH` | HKUSTAudio/xcodec2 | XCodec2 模型路径 |
| `WHISPER_MODEL_PATH` | Systran/faster-whisper-large-v3 | Whisper 模型路径 |

### 配置文件

从模板创建 `.env`：
```bash
cp .env.example .env
```

编辑 `.env`：
```bash
# GPU 配置
NVIDIA_VISIBLE_DEVICES=0
GPU_IDLE_TIMEOUT=600

# 端口配置
UI_PORT=7860
API_PORT=7861

# HuggingFace 配置
HF_ENDPOINT=https://hf-mirror.com
# HF_TOKEN=your_token_here

# 模型路径（可选，使用本地模型）
# LLASA_MODEL_PATH=/path/to/Llasa-8B
# XCODEC_MODEL_PATH=/path/to/xcodec2
```

## 📖 使用方法

### Web UI

1. 打开 http://localhost:7860
2. 上传参考音频（15-20 秒，WAV 格式）
3. 点击"自动转录"或手动输入参考文本
4. 输入要生成的目标文本
5. 点击"生成语音"

### REST API

**健康检查：**
```bash
curl http://localhost:7861/health
```

**生成语音：**
```bash
curl -X POST http://localhost:7861/api/tts \
  -F "audio=@reference.wav" \
  -F "ref_text=参考音频文本" \
  -F "target_text=要生成的文本" \
  --output generated.wav
```

**GPU 状态：**
```bash
curl http://localhost:7861/api/gpu/status
```

**手动卸载：**
```bash
curl -X POST http://localhost:7861/api/gpu/offload
```

### MCP（模型上下文协议）

运行 MCP 服务器：
```bash
docker exec -it llasa-tts-8b-webui python mcp_server.py
```

或在宿主机运行：
```bash
python mcp_server.py
```

可用工具：
- `generate_speech()` - 生成语音
- `transcribe_audio()` - 转录音频
- `get_gpu_status()` - 获取 GPU 状态
- `offload_gpu()` - 卸载 GPU 显存
- `release_gpu()` - 完全释放 GPU

## 📚 API 文档

### 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/gpu/status` | 获取 GPU 状态 |
| POST | `/api/gpu/offload` | 卸载模型到 CPU |
| POST | `/api/gpu/release` | 释放所有模型 |
| POST | `/api/transcribe` | 转录音频（ASR） |
| POST | `/api/tts` | 生成语音（TTS） |
| GET | `/apidocs` | Swagger 文档 |

### API 示例

查看 [API 文档](http://localhost:7861/apidocs) 获取交互式示例。

## 🛠️ 技术栈

### 核心技术
- **PyTorch 2.6.0** - 深度学习框架
- **Transformers 4.45.2** - 模型加载
- **Gradio 4.0+** - Web UI
- **Flask 3.0.0** - REST API
- **FastMCP** - MCP 服务器

### 模型
- **Llasa-8B** - 语音生成（~17GB）
- **XCodec2** - 音频编解码（~3GB）
- **Faster-Whisper** - 语音识别（~3GB，CPU）

### 部署
- **Docker** - 容器化
- **Docker Compose** - 编排
- **NVIDIA Docker** - GPU 支持

## 📊 性能指标

### GPU 显存使用

| 阶段 | 传统方式 | 智能管理 | 节省 |
|------|---------|---------|------|
| 启动 | 24 GB | 0 GB | 100% |
| 运行中 | 24 GB | 24 GB | 0% |
| 空闲时 | 24 GB | < 1 GB | **96%** |

### 加载时间（RTX 4090）

- 首次加载：20-30 秒
- CPU → GPU：2-5 秒
- GPU → CPU：2 秒
- 完全释放：1 秒

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支（`git checkout -b feature/AmazingFeature`）
3. 提交您的更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 打开一个 Pull Request

## 📝 更新日志

### v1.0.0 (2025-12-06)
- ✨ 初始版本发布
- 🚀 智能 GPU 显存管理
- 🎨 三种访问模式（UI + API + MCP）
- 🔄 自动 GPU 选择
- 🐳 Docker 部署

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

**注意**：模型和依赖项有各自的许可证：
- Llasa-8B：[官方许可证](https://huggingface.co/HKUSTAudio/Llasa-8B)
- XCodec2：[官方许可证](https://huggingface.co/HKUSTAudio/xcodec2)
- Faster-Whisper：Apache 2.0

## 🙏 致谢

- 原始项目：[HKUSTAudio/Llasa-8B](https://huggingface.co/HKUSTAudio/Llasa-8B)
- 感谢所有贡献者和开源社区

## 📞 联系与支持

- 📧 问题反馈：[GitHub Issues](https://github.com/yourusername/llasa-tts-8b-webui/issues)
- 💬 讨论区：[GitHub Discussions](https://github.com/yourusername/llasa-tts-8b-webui/discussions)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/llasa-tts-8b-webui&type=Date)](https://star-history.com/#yourusername/llasa-tts-8b-webui)

## 📱 关注公众号

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

---

<p align="center">用 ❤️ 制作 by Llasa-TTS-8B 团队</p>
