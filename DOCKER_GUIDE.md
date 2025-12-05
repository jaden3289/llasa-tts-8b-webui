# Llasa-8B TTS WebUI Docker 部署指南（智能 GPU 管理版）

## ✨ 新特性

本项目现已支持 **GPU 智能显存管理**：

- ✅ **自动选择最空闲的 GPU** - 启动时自动检测并使用显存占用最少的 GPU
- ✅ **懒加载** - 首次请求时才加载模型，加快启动速度
- ✅ **即用即卸** - 任务完成后立即释放 GPU 显存（从 24GB 降至 < 1GB）
- ✅ **三种访问模式** - Web UI + REST API + MCP（Model Context Protocol）

详见：[GPU 管理文档](./GPU_MANAGEMENT.md) | [MCP 使用指南](./MCP_GUIDE.md)

## 📋 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support (至少 24GB 显存)
- **软件依赖**:
  - Docker >= 20.10
  - Docker Compose >= 2.0
  - NVIDIA Docker Runtime (nvidia-docker2)
  - NVIDIA Driver >= 525.60.13
  - CUDA >= 12.1

## 🚀 快速开始

### 1. 克隆或下载项目

```bash
git clone <your-repo-url>
cd Llasa-TTS-8B-WebUI-Demo
```

### 2. 赋予脚本执行权限

```bash
chmod +x start.sh stop.sh
```

### 3. 一键启动

```bash
./start.sh
```

**启动脚本会自动：**
- ✅ 检查 Docker 和 NVIDIA Docker 环境
- ✅ 检测所有 GPU 并显示状态
- ✅ **自动选择显存占用最少的 GPU**
- ✅ 创建 `.env` 配置文件
- ✅ 创建必要的目录（models_cache, outputs, temp）
- ✅ 构建 Docker 镜像
- ✅ 启动容器

### 4. 访问服务

启动成功后，可以通过以下方式访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| **Web UI** | http://localhost:7860 | Gradio 可视化界面 |
| **REST API** | http://localhost:7861 | RESTful API 接口 |
| **API 文档** | http://localhost:7861/apidocs | Swagger API 文档 |
| **MCP** | 通过 MCP 客户端连接 | 程序化访问接口 |

首次启动会自动下载模型（约 20GB），请耐心等待。

## 🛠️ 手动操作

### 构建镜像

```bash
docker-compose build
```

### 启动容器

```bash
docker-compose up -d
```

### 查看日志

```bash
# 实时查看日志
docker-compose logs -f

# 查看最近的日志
docker-compose logs --tail=100
```

### 停止服务

```bash
# 使用脚本
./stop.sh

# 或手动执行
docker-compose down
```

### 重启服务

```bash
docker-compose restart
```

### 进入容器

```bash
docker-compose exec llasa-tts-webui bash
```

## ⚙️ 配置说明

### GPU 设置

默认配置使用 **GPU 1 和 2**（因为您提到这两个 GPU 比较空闲）。

如果需要修改使用的 GPU，编辑 `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['1', '2']  # 修改这里，例如改为 ['0', '1']
          capabilities: [gpu]
```

或者在启动时通过环境变量指定:

```bash
CUDA_VISIBLE_DEVICES=0,1 docker-compose up -d
```

### HuggingFace Token 配置

如果需要从 HuggingFace 下载模型，需要设置 token:

1. 在 HuggingFace 网站获取 token: https://huggingface.co/settings/tokens
2. 编辑 `docker-compose.yml`，取消注释并设置:

```yaml
environment:
  - HF_TOKEN=your_huggingface_token_here
```

### 使用本地模型

如果已经下载了模型到本地，可以挂载到容器：

1. 下载模型到本地目录，例如: `/home/neo/models/Llasa-8B`
2. 修改 `docker-compose.yml`:

```yaml
volumes:
  - /home/neo/models:/models
```

3. 修改 `app.py` 中的模型路径:

```python
llasa_8b = '/models/Llasa-8B'
model_path = "/models/xcodec2"
fastwhisper_path = "/models/faster-whisper-large-v3"
```

## 📊 资源监控

### 查看 GPU 使用情况

在容器内执行:

```bash
docker-compose exec llasa-tts-webui nvidia-smi
```

在主机上执行:

```bash
watch -n 1 nvidia-smi
```

### 查看容器资源使用

```bash
docker stats llasa-tts-8b-webui
```

## 🐛 故障排查

### 问题 1: 容器启动失败

**检查日志**:
```bash
docker-compose logs llasa-tts-webui
```

**常见原因**:
- GPU 不可用或驱动问题
- 显存不足
- 端口 7860 被占用

### 问题 2: 无法访问 WebUI

**检查容器状态**:
```bash
docker-compose ps
```

**检查端口**:
```bash
netstat -tulpn | grep 7860
```

### 问题 3: 模型下载缓慢

**解决方案**:
1. 使用 HF-Mirror (已默认配置)
2. 手动下载模型到本地并挂载
3. 使用代理

### 问题 4: 显存不足 (OOM)

**解决方案**:
1. 确保使用的 GPU 有足够显存 (至少 24GB)
2. 关闭其他占用 GPU 的程序
3. 考虑使用模型量化 (需要修改代码)

### 问题 5: NVIDIA Docker 运行时未找到

**安装 nvidia-docker**:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🔧 高级配置

### 修改端口

编辑 `docker-compose.yml`:

```yaml
ports:
  - "8080:7860"  # 将 7860 改为你想要的端口
```

### 添加资源限制

编辑 `docker-compose.yml`，取消注释:

```yaml
mem_limit: 32g       # 限制内存使用
shm_size: 8g         # 增加共享内存
```

### 配置自动重启

已默认配置 `restart: unless-stopped`，容器会在系统重启后自动启动。

### 使用外部数据库或缓存

可以通过 volumes 挂载外部目录来持久化数据。

## 📚 模型信息

### 所需模型

1. **Llasa-8B** (~17GB 显存)
   - 地址: https://huggingface.co/HKUSTAudio/Llasa-8B

2. **XCodec2** (~3GB 显存)
   - 地址: https://huggingface.co/HKUSTAudio/xcodec2

3. **Faster-Whisper-Large-V3** (~3GB 显存，默认 CPU)
   - 地址: https://huggingface.co/Systran/faster-whisper-large-v3

### 模型缓存位置

所有下载的模型会缓存到 `./models_cache` 目录，下次启动时会直接使用，无需重新下载。

## 🌐 网络配置

### 允许外网访问

1. 确保防火墙允许 7860 端口
2. WebUI 已配置监听 `0.0.0.0`
3. 访问地址: `http://your_server_ip:7860`

### 安全建议

如果暴露到公网，建议：
- 使用 Nginx 反向代理
- 配置 HTTPS
- 添加身份认证
- 使用防火墙限制访问 IP

## 📝 性能优化

### 1. 使用更快的镜像源

已在 Dockerfile 中配置了清华源和 HF-Mirror。

### 2. 启用 GPU 加速 Whisper

修改 `app.py`:

```python
fastwhisper_model = WhisperModel(fastwhisper_path, device="cuda")  # 改为 cuda
```

**注意**: 这会额外占用 3GB 显存

### 3. 批量处理

可以修改代码支持批量生成，提高吞吐量。

## 🆘 获取帮助

- **查看日志**: `docker-compose logs -f`
- **官方模型**: https://huggingface.co/HKUSTAudio/Llasa-8B
- **问题反馈**: https://github.com/HKUSTAudio/Llasa/issues

## 📄 License

本项目的代码 (app.py) 使用 MIT License。
模型和其他依赖请参考其官方 License。
