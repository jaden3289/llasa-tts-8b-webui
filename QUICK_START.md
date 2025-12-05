# 🚀 Llasa-8B TTS WebUI 快速启动指南

## 📦 项目概述

本项目是 **Llasa-8B** 语音合成模型的 Docker 化应用，支持：

- ✅ **开箱即用**：Docker 一键部署，无需复杂环境配置
- ✅ **智能 GPU 管理**：自动选择最空闲 GPU + 懒加载 + 即用即卸
- ✅ **三种访问模式**：Web UI + REST API + MCP
- ✅ **现代化 UI**：美观、易用的 Gradio 界面
- ✅ **多语言支持**：中英文及混合语音生成
- ✅ **语音克隆**：基于参考音频生成相似音色的语音

## 🎯 一键启动（推荐）

```bash
# 进入项目目录
cd /home/neo/upload/Llasa-TTS-8B-WebUI-Demo

# 启动服务
./start.sh
```

启动后可通过以下方式访问：

| 服务 | 地址 | 说明 |
|------|------|------|
| **Web UI** | http://localhost:7860 | 可视化界面 |
| **REST API** | http://localhost:7861 | RESTful API |
| **API 文档** | http://localhost:7861/apidocs | Swagger 文档 |
| **MCP** | 通过 MCP 客户端 | 程序化接口 |

## 📊 智能 GPU 管理

### 自动选择最空闲 GPU

`start.sh` 会自动检测所有 GPU，并选择显存占用最少的 GPU：

```bash
# 自动检测示例输出
检测到 4 个 GPU
当前 GPU 状态：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU 0: NVIDIA RTX 4090
  显存: 15234 MiB / 24564 MiB
  利用率: 80 %
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU 1: NVIDIA RTX 4090
  显存: 456 MiB / 24564 MiB  ← 最空闲
  利用率: 2 %
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

已选择 GPU 1 (显存使用最少)
```

### GPU 显存智能管理

模型采用 **懒加载 + 即用即卸** 策略：

- ⏳ **首次加载**：20-30 秒（从磁盘）
- 🚀 **处理任务**：视任务而定（模型在 GPU 上）
- 📥 **自动卸载**：2 秒（转移到 CPU，释放显存）
- ⚡ **快速恢复**：2-5 秒（从 CPU 恢复到 GPU）

**好处**：
- GPU 显存从 24GB 降至 < 1GB（空闲时）
- 其他容器可以共享 GPU 资源
- 多个服务可以在同一 GPU 上运行

详见：[GPU 管理文档](./GPU_MANAGEMENT.md)

## 🎨 UI 界面特点

### 重新设计的现代化界面

1. **渐变色标题**：紫色渐变，视觉效果更佳
2. **分步骤引导**：清晰的 3 步操作流程
3. **信息提示框**：蓝色信息框提供使用提示
4. **折叠面板**：高级设置和使用说明可折叠，界面更简洁
5. **Emoji 图标**：每个功能都有对应的图标，更直观
6. **响应式布局**：自适应不同屏幕尺寸

### 主要功能区

- **步骤 1**: 上传参考音频 + 自动转录
- **步骤 2**: 输入要生成的文本
- **步骤 3**: 查看生成结果
- **高级设置**: 温度、采样率、Top-K/P 等参数调节

## 📁 项目结构

```
Llasa-TTS-8B-WebUI-Demo/
├── app.py                 # 主应用程序（已优化 UI 和 GPU 设置）
├── requirements.txt       # Python 依赖（已完善）
├── Dockerfile            # Docker 镜像定义（使用 PyTorch CUDA 镜像）
├── docker-compose.yml    # Docker Compose 配置（GPU 1, 2）
├── .dockerignore         # Docker 构建忽略文件
├── start.sh              # 一键启动脚本
├── stop.sh               # 停止服务脚本
├── README.md             # 项目说明文档
├── DOCKER_GUIDE.md       # Docker 详细部署指南
├── QUICK_START.md        # 快速启动指南（本文件）
└── models_cache/         # 模型缓存目录（自动创建）
```

## 🔧 常用命令

### 查看服务状态

```bash
docker-compose ps
```

### 查看实时日志

```bash
docker-compose logs -f
```

### 重启服务

```bash
docker-compose restart
```

### 停止服务

```bash
./stop.sh
# 或
docker-compose down
```

### 查看 GPU 使用情况

```bash
# 主机上
watch -n 1 nvidia-smi

# 容器内
docker-compose exec llasa-tts-webui nvidia-smi
```

## 📝 使用流程

### 1. 准备参考音频

- 格式: .wav
- 时长: 15-20 秒（建议）
- 质量: 清晰、无背景噪音
- 内容: 中文或英文朗读

### 2. 上传并转录

1. 上传参考音频
2. 点击"自动转录"按钮（或手动输入文本）
3. 确认转录结果与音频内容一致

### 3. 输入目标文本

- 在"要生成的文字内容"框中输入你想要生成的文本
- 支持中英文混合
- 最多 300 字符

### 4. 生成语音

- 点击"生成语音"按钮
- 等待处理（约 20-30 秒）
- 下载或播放生成的音频

## ⚙️ 参数调优

### 推荐参数（默认值）

- **Temperature**: 1.0（平衡稳定性和多样性）
- **Top K**: 50
- **Top P**: 0.9
- **Penalty**: 1.2（减少重复）
- **Sample Rate**: 24000 Hz（推荐）

### 参数效果

- **Temperature ↑**: 生成更有变化，但可能不稳定
- **Temperature ↓**: 生成更稳定，但可能单调
- **Penalty ↑**: 减少重复内容
- **Sample Rate ↑**: 音质更好，但生成速度稍慢

## 🐛 常见问题

### Q1: 启动时显示 "GPU 不可用"

**解决方案**:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Q2: 首次启动很慢

**原因**: 正在下载模型（约 20GB）

**查看进度**:
```bash
docker-compose logs -f llasa-tts-webui
```

### Q3: 端口 7860 已被占用

**解决方案**: 修改 `docker-compose.yml` 中的端口映射
```yaml
ports:
  - "8080:7860"  # 改为其他端口
```

### Q4: 生成的语音质量不好

**可能原因**:
- 参考音频质量差
- 参考文本不准确
- 生成文本与参考风格差异大

**解决方案**:
- 使用高质量参考音频
- 确保转录准确
- 调整参数（降低 Temperature）

## 📊 资源需求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | 24GB 显存 | 2x 24GB GPU |
| 内存 | 16GB | 32GB |
| 硬盘 | 50GB | 100GB SSD |
| CPU | 4 核 | 8 核 |

### 显存占用

- Llasa-8B: ~17GB
- XCodec2: ~3GB
- FastWhisper (CPU): ~0GB
- FastWhisper (GPU): ~3GB
- **总计**: ~20-23GB

## 🌐 外网访问

如果需要从外网访问：

1. **开放端口**:
```bash
sudo ufw allow 7860
```

2. **访问地址**:
```
http://your_server_ip:7860
```

3. **安全建议**:
- 使用 Nginx 反向代理
- 配置 HTTPS
- 添加身份认证
- 使用防火墙限制访问 IP

## 🔄 更新项目

```bash
# 停止服务
./stop.sh

# 拉取最新代码
git pull

# 重新构建并启动
./start.sh
```

## 📚 更多文档

- **详细部署指南**: [DOCKER_GUIDE.md](./DOCKER_GUIDE.md)
- **项目说明**: [README.md](./README.md)
- **官方模型**: https://huggingface.co/HKUSTAudio/Llasa-8B

## 🆘 获取帮助

如有问题，请：

1. 查看日志: `docker-compose logs -f`
2. 检查 GPU: `nvidia-smi`
3. 阅读文档: [DOCKER_GUIDE.md](./DOCKER_GUIDE.md)
4. 官方 Issues: https://huggingface.co/HKUSTAudio/Llasa-8B/discussions

---

**祝使用愉快！** 🎉
