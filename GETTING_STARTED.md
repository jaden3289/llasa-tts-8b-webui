# 🚀 Llasa-TTS-8B 快速上手指南

## 📋 前置要求

在开始之前，请确保您的系统满足以下要求：

### 硬件要求
- ✅ NVIDIA GPU（至少 24GB 显存）
- ✅ 至少 32GB 系统内存
- ✅ 至少 50GB 可用磁盘空间（用于模型缓存）

### 软件要求
- ✅ Linux 系统（推荐 Ubuntu 20.04+）
- ✅ Docker（20.10+）
- ✅ Docker Compose（1.29+ 或 Docker Compose V2）
- ✅ NVIDIA 驱动（推荐 525+）
- ✅ nvidia-docker（NVIDIA Container Toolkit）

---

## 🎯 5 分钟快速启动

### 步骤 1: 验证环境

```bash
# 进入项目目录
cd Llasa-TTS-8B-WebUI-Demo

# 运行验证脚本
./verify_project.sh
```

**预期输出**：
```
✅ 验证通过！项目已准备就绪
```

如果看到错误，请根据提示安装缺失的依赖。

---

### 步骤 2: 启动服务

```bash
# 一键启动（自动选择最空闲的 GPU）
./start.sh
```

**启动过程**：
1. ✅ 检查 Docker 环境
2. ✅ 检测并选择最空闲的 GPU
3. ✅ 创建/更新配置文件
4. ✅ 构建 Docker 镜像（首次需要 5-10 分钟）
5. ✅ 启动容器
6. ✅ 显示访问信息

**预期输出**：
```
✅ Llasa-8B TTS WebUI 已成功启动!

📊 服务访问地址：
   • Web UI:  http://localhost:7860
   • API:     http://localhost:7861
   • API Doc: http://localhost:7861/apidocs

🎮 GPU 信息：
   • 使用 GPU: 0 (NVIDIA GeForce RTX 4090)
```

---

### 步骤 3: 访问服务

#### 方式 1: Web UI（推荐新手）

1. 打开浏览器访问：http://localhost:7860
2. 上传参考音频（15-20 秒的 .wav 文件）
3. 点击「自动转录」或手动输入参考文本
4. 输入要生成的文字内容
5. 点击「生成语音」

**提示**：
- 首次生成需要下载模型（约 20GB），请耐心等待
- 可以通过 `docker logs -f llasa-tts-8b-webui` 查看下载进度

#### 方式 2: REST API（推荐开发者）

```bash
# 健康检查
curl http://localhost:7861/health

# 查看 API 文档
open http://localhost:7861/apidocs

# 生成语音（需要准备测试音频）
curl -X POST http://localhost:7861/api/tts \
  -F "audio=@reference.wav" \
  -F "ref_text=这是参考音频的文本" \
  -F "target_text=你好，欢迎使用 Llasa TTS" \
  --output generated.wav
```

#### 方式 3: MCP（推荐高级用户）

```bash
# 在容器内运行 MCP 服务器
docker exec -it llasa-tts-8b-webui python mcp_server.py
```

详见：[MCP 使用指南](./MCP_GUIDE.md)

---

### 步骤 4: 运行测试（可选）

```bash
# 运行完整测试
./test_all.sh
```

**测试内容**：
- ✅ Docker 环境测试
- ✅ GPU 状态测试
- ✅ 服务健康检查
- ✅ GPU 管理 API 测试
- ✅ 显存占用测试

---

## 🎨 使用示例

### 示例 1: 基础语音生成

**场景**：使用参考音频生成新的语音

1. 准备参考音频：
   - 格式：WAV
   - 时长：15-20 秒
   - 质量：清晰、无背景噪音

2. 在 Web UI 中：
   - 上传参考音频
   - 自动转录或手动输入文本
   - 输入要生成的内容（最多 300 字）
   - 点击生成

3. 等待生成（约 20-30 秒）

4. 下载生成的音频

### 示例 2: API 批量生成

```bash
#!/bin/bash

# 批量生成脚本
for text in "你好" "欢迎" "谢谢"; do
    curl -X POST http://localhost:7861/api/tts \
      -F "audio=@reference.wav" \
      -F "ref_text=参考文本" \
      -F "target_text=${text}" \
      --output "${text}.wav"
    
    echo "生成完成: ${text}.wav"
done
```

### 示例 3: 参数调优

**调整参数以获得不同效果**：

| 参数 | 默认值 | 效果 |
|------|--------|------|
| temperature | 1.0 | 越低越稳定，越高越多样 |
| top_k | 50 | 控制采样范围 |
| top_p | 0.9 | Nucleus 采样阈值 |
| penalty | 1.2 | 重复惩罚系数 |

**示例**：
```bash
# 更稳定的生成
curl -X POST http://localhost:7861/api/tts \
  -F "audio=@reference.wav" \
  -F "ref_text=参考文本" \
  -F "target_text=目标文本" \
  -F "temperature=0.7" \
  -F "penalty=1.5"
```

---

## 🔧 常见问题

### Q1: 首次启动很慢？

**A**: 首次启动需要：
1. 构建 Docker 镜像（5-10 分钟）
2. 下载模型（约 20GB，10-30 分钟）

**解决方案**：
- 耐心等待
- 查看日志：`docker logs -f llasa-tts-8b-webui`
- 使用国内镜像：已配置 hf-mirror

### Q2: GPU 显存不足？

**A**: 本项目使用智能 GPU 管理：
- 空闲时显存 < 1GB
- 运行时需要约 24GB

**解决方案**：
- 确保 GPU 至少有 24GB 显存
- 关闭其他占用 GPU 的程序
- 手动释放显存：`curl -X POST http://localhost:7861/api/gpu/release`

### Q3: 生成的语音质量不好？

**A**: 可能的原因：
1. 参考音频质量差
2. 参考文本不准确
3. 参数设置不当

**解决方案**：
1. 使用高质量的参考音频（清晰、无噪音）
2. 确保参考文本与音频内容一致
3. 调整参数（降低 temperature，增加 penalty）

### Q4: 如何查看日志？

```bash
# 实时日志
docker logs -f llasa-tts-8b-webui

# 最近 100 行
docker logs --tail 100 llasa-tts-8b-webui

# 保存日志到文件
docker logs llasa-tts-8b-webui > logs.txt
```

### Q5: 如何重启服务？

```bash
# 方式 1: 使用脚本
./stop.sh
./start.sh

# 方式 2: 使用 docker-compose
docker-compose restart

# 方式 3: 重启容器
docker restart llasa-tts-8b-webui
```

### Q6: 如何更新代码？

```bash
# 1. 停止服务
./stop.sh

# 2. 拉取最新代码
git pull

# 3. 重新构建镜像
docker-compose build

# 4. 启动服务
./start.sh
```

---

## 📊 监控和管理

### GPU 状态监控

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 通过 API 查看状态
curl http://localhost:7861/api/gpu/status | jq
```

### 手动 GPU 管理

```bash
# 卸载模型到 CPU（释放显存）
curl -X POST http://localhost:7861/api/gpu/offload

# 完全释放模型
curl -X POST http://localhost:7861/api/gpu/release

# 查看状态
curl http://localhost:7861/api/gpu/status
```

### 容器管理

```bash
# 查看容器状态
docker ps

# 查看资源使用
docker stats llasa-tts-8b-webui

# 进入容器
docker exec -it llasa-tts-8b-webui bash

# 查看容器详情
docker inspect llasa-tts-8b-webui
```

---

## 🎓 进阶使用

### 1. 使用本地模型

如果已经下载了模型到本地：

```bash
# 编辑 .env 文件
LLASA_MODEL_PATH=/path/to/Llasa-8B
XCODEC_MODEL_PATH=/path/to/xcodec2
WHISPER_MODEL_PATH=/path/to/faster-whisper-large-v3

# 在 docker-compose.yml 中挂载目录
volumes:
  - /path/to/models:/models
```

### 2. 多 GPU 部署

```bash
# 启动多个实例，使用不同的 GPU
NVIDIA_VISIBLE_DEVICES=0 UI_PORT=7860 API_PORT=7861 docker-compose up -d
NVIDIA_VISIBLE_DEVICES=1 UI_PORT=7870 API_PORT=7871 docker-compose up -d
```

### 3. 自定义配置

编辑 `.env` 文件：

```bash
# GPU 配置
GPU_IDLE_TIMEOUT=300  # 5 分钟后自动卸载

# 端口配置
UI_PORT=8080
API_PORT=8081

# 资源限制
MEM_LIMIT=64g
SHM_SIZE=16g
```

---

## 📚 更多文档

- [架构说明](./ARCHITECTURE.md) - 了解系统架构
- [GPU 管理详解](./GPU_MANAGEMENT.md) - 深入了解 GPU 管理
- [MCP 使用指南](./MCP_GUIDE.md) - MCP 接口使用
- [Docker 部署指南](./DOCKER_GUIDE.md) - 详细部署说明
- [项目总结](./PROJECT_SUMMARY.md) - 完整功能列表

---

## 🆘 获取帮助

如果遇到问题：

1. 查看日志：`docker logs -f llasa-tts-8b-webui`
2. 运行测试：`./test_all.sh`
3. 查看文档：阅读相关文档
4. 提交 Issue：在 GitHub 上提交问题

---

## 🎉 开始使用

现在您已经准备好使用 Llasa-TTS-8B 了！

```bash
# 启动服务
./start.sh

# 访问 Web UI
open http://localhost:7860

# 开始生成语音！
```

祝使用愉快！🎙️
