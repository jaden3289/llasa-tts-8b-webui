# Llasa-TTS-8B 架构说明

## 📐 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker 容器                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              main.py (统一启动入口)                     │    │
│  └────────────────────────────────────────────────────────┘    │
│           │                          │                          │
│           ▼                          ▼                          │
│  ┌─────────────────┐        ┌─────────────────┐               │
│  │   app.py        │        │  api_server.py  │               │
│  │  (Gradio UI)    │        │  (Flask API)    │               │
│  │  端口: 7860     │        │  端口: 7861     │               │
│  └─────────────────┘        └─────────────────┘               │
│           │                          │                          │
│           └──────────┬───────────────┘                          │
│                      ▼                                          │
│         ┌─────────────────────────┐                            │
│         │   gpu_manager.py        │                            │
│         │  (GPU 资源管理器)       │                            │
│         └─────────────────────────┘                            │
│                      │                                          │
│         ┌────────────┴────────────┐                            │
│         ▼                         ▼                            │
│  ┌─────────────┐          ┌─────────────┐                     │
│  │  Llasa-8B   │          │  XCodec2    │                     │
│  │  (语音生成)  │          │  (编解码)    │                     │
│  └─────────────┘          └─────────────┘                     │
│         │                         │                            │
│         └────────────┬────────────┘                            │
│                      ▼                                          │
│              ┌──────────────┐                                  │
│              │   GPU / CPU  │                                  │
│              └──────────────┘                                  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              mcp_server.py (可选单独运行)               │    │
│  │              (Model Context Protocol)                   │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 核心组件

### 1. main.py - 统一启动入口

**功能**：
- 同时启动 Gradio UI 和 Flask API
- 管理多线程启动
- 统一日志输出

**启动流程**：
```python
1. 创建必要目录 (outputs, temp)
2. 后台线程启动 API 服务器 (端口 7861)
3. 主线程启动 Gradio UI (端口 7860)
```

### 2. app.py - Gradio Web UI

**功能**：
- 提供用户友好的 Web 界面
- 支持音频上传和参数调整
- 实时进度显示
- 自动转录功能（Faster-Whisper）

**特点**：
- 现代化 UI 设计
- 响应式布局
- 详细的使用说明
- 参数分组展示

### 3. api_server.py - REST API 服务器

**功能**：
- 提供 RESTful API 接口
- Swagger 文档自动生成
- GPU 状态管理接口
- 音频转录和 TTS 接口

**主要端点**：
```
GET  /health              - 健康检查
GET  /api/gpu/status      - GPU 状态
POST /api/gpu/offload     - 卸载模型到 CPU
POST /api/gpu/release     - 完全释放模型
POST /api/transcribe      - 音频转录
POST /api/tts             - 语音生成
GET  /apidocs             - Swagger 文档
```

### 4. gpu_manager.py - GPU 资源管理器 ⭐

**核心功能**：

#### 单模型管理器 (GPUResourceManager)
```python
class GPUResourceManager:
    - get_model()         # 懒加载模型
    - force_offload()     # 立即卸载到 CPU
    - force_release()     # 完全释放
    - get_status()        # 获取状态
```

#### 多模型管理器 (MultiModelGPUManager)
```python
class MultiModelGPUManager:
    - register_model()    # 注册模型
    - offload_all()       # 卸载所有模型
    - release_all()       # 释放所有模型
    - get_all_status()    # 获取所有状态
```

**状态转换**：
```
未加载 ──首次(20-30s)──> GPU ──任务完成(2s)──> CPU ──新请求(2-5s)──> GPU
  ↑                                                    ↓
  └──────────────超时/手动释放(1s)─────────────────────┘
```

### 5. mcp_server.py - MCP 服务器

**功能**：
- 提供 Model Context Protocol 接口
- 程序化访问 TTS 功能
- 共享 GPU 管理器

**可用工具**：
```python
- generate_speech()      # 生成语音
- transcribe_audio()     # 转录音频
- get_gpu_status()       # 获取 GPU 状态
- offload_gpu()          # 卸载 GPU
- release_gpu()          # 释放 GPU
- update_gpu_timeout()   # 更新超时
```

## 🔄 工作流程

### TTS 生成流程

```
1. 用户输入
   ├─ 参考音频 (WAV)
   ├─ 参考文本
   └─ 目标文本

2. 音频编码 (XCodec2)
   ├─ 加载 XCodec2 模型到 GPU
   ├─ 编码参考音频
   └─ 立即卸载到 CPU

3. 语音生成 (Llasa-8B)
   ├─ 加载 Llasa-8B 模型到 GPU
   ├─ 生成语音 token
   └─ 立即卸载到 CPU

4. 音频解码 (XCodec2)
   ├─ 重新加载 XCodec2 到 GPU
   ├─ 解码语音 token
   └─ 立即卸载到 CPU

5. 输出结果
   └─ 保存生成的音频文件
```

### GPU 管理流程

```
请求到达
    ↓
检查模型位置
    ├─ 在 GPU 上 → 直接使用
    ├─ 在 CPU 上 → 快速转移到 GPU (2-5s)
    └─ 未加载   → 从磁盘加载 (20-30s)
    ↓
执行任务
    ↓
立即卸载到 CPU (2s)
    ↓
释放 GPU 显存
    ↓
等待下次请求
```

## 📊 显存管理

### 显存占用对比

| 阶段 | 传统方式 | 智能管理 | 节省 |
|------|---------|---------|------|
| 启动 | 24 GB | 0 GB | 100% |
| 运行中 | 24 GB | 24 GB | 0% |
| 空闲时 | 24 GB | < 1 GB | **96%** |

### 模型显存占用

| 模型 | 显存 | 位置 |
|------|------|------|
| Llasa-8B | ~17 GB | GPU (使用时) / CPU (空闲时) |
| XCodec2 | ~3 GB | GPU (使用时) / CPU (空闲时) |
| Faster-Whisper | ~3 GB | CPU (固定) |

## 🚀 部署架构

### Docker 容器配置

```yaml
services:
  llasa-tts-webui:
    # GPU 配置
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['${NVIDIA_VISIBLE_DEVICES}']
              capabilities: [gpu]
    
    # 端口映射
    ports:
      - "7860:7860"  # Gradio UI
      - "7861:7861"  # REST API
      - "7862:7862"  # MCP (可选)
    
    # 目录挂载
    volumes:
      - ./models_cache:/root/.cache/huggingface
      - ./outputs:/app/outputs
      - ./temp:/app/temp
```

### 自动 GPU 选择

```bash
# start.sh 自动选择最空闲的 GPU
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used \
         --format=csv,noheader,nounits | \
         sort -t',' -k2 -n | head -1 | cut -d',' -f1)

export NVIDIA_VISIBLE_DEVICES=$GPU_ID
```

## 🔌 访问方式

### 1. Web UI (Gradio)

```
URL: http://localhost:7860
特点：
  - 用户友好的界面
  - 实时进度显示
  - 参数可视化调整
  - 音频预览和下载
```

### 2. REST API (Flask)

```
URL: http://localhost:7861
文档: http://localhost:7861/apidocs

示例：
curl -X POST http://localhost:7861/api/tts \
  -F "audio=@reference.wav" \
  -F "ref_text=参考文本" \
  -F "target_text=目标文本"
```

### 3. MCP (Model Context Protocol)

```
运行方式：
docker exec -it llasa-tts-8b-webui python mcp_server.py

特点：
  - 程序化访问
  - 共享 GPU 管理器
  - 完整的工具集
```

## 🛠️ 维护和监控

### 查看日志

```bash
# 实时日志
docker logs -f llasa-tts-8b-webui

# 最近 100 行
docker logs --tail 100 llasa-tts-8b-webui
```

### GPU 监控

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看显存使用
nvidia-smi --query-gpu=memory.used --format=csv
```

### API 状态检查

```bash
# 健康检查
curl http://localhost:7861/health

# GPU 状态
curl http://localhost:7861/api/gpu/status | jq
```

## 📝 配置文件

### .env 环境变量

```bash
# GPU 配置
NVIDIA_VISIBLE_DEVICES=0
GPU_IDLE_TIMEOUT=600

# 端口配置
UI_PORT=7860
API_PORT=7861

# 模型路径
LLASA_MODEL_PATH=HKUSTAudio/Llasa-8B
XCODEC_MODEL_PATH=HKUSTAudio/xcodec2
WHISPER_MODEL_PATH=Systran/faster-whisper-large-v3
```

## 🔒 安全考虑

1. **端口绑定**：默认绑定到 0.0.0.0，允许外部访问
2. **API 认证**：可通过 API_KEY 环境变量启用
3. **资源限制**：Docker 配置了内存和显存限制
4. **日志管理**：自动轮转，限制大小

## 🚦 性能优化

1. **模型缓存**：首次下载后缓存到本地
2. **GPU 复用**：多个请求共享同一 GPU
3. **异步处理**：API 支持异步任务
4. **显存优化**：自动卸载空闲模型

## 📚 相关文档

- [GPU 管理详解](./GPU_MANAGEMENT.md)
- [MCP 使用指南](./MCP_GUIDE.md)
- [Docker 部署指南](./DOCKER_GUIDE.md)
- [快速开始](./QUICK_START.md)
