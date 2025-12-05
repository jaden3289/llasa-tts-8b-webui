# MCP (Model Context Protocol) 使用指南

## 📖 什么是 MCP?

**Model Context Protocol (MCP)** 是一个程序化访问接口，允许其他应用程序或 AI 助手直接调用本项目的功能。

与传统的 REST API 相比，MCP 提供：
- ✅ 更简洁的工具定义
- ✅ 内置类型检查
- ✅ 更好的错误处理
- ✅ 适合 AI Agent 集成

## 🚀 快速开始

### 1. 启动 MCP 服务器

MCP 服务器已集成在 Docker 容器中：

```bash
# 启动所有服务（包括 MCP）
./start.sh

# MCP 服务器将在容器内运行
```

### 2. 配置 MCP 客户端

创建 MCP 配置文件（例如 `.mcp/config.json`）：

```json
{
  "mcpServers": {
    "llasa-tts": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "llasa-tts-8b-webui",
        "python",
        "/app/mcp_server.py"
      ],
      "env": {
        "GPU_IDLE_TIMEOUT": "600"
      }
    }
  }
}
```

### 3. 使用 MCP 工具

在支持 MCP 的客户端中使用工具：

```python
from mcp import ClientSession

async with ClientSession() as session:
    # 调用工具
    result = await session.call_tool("generate_speech", {
        "audio_path": "/path/to/reference.wav",
        "ref_text": "这是参考音频的文本",
        "target_text": "这是要生成的文本"
    })

    print(result)
```

## 🛠️ 可用工具

### 1. generate_speech - 生成语音

**描述**：根据参考音频生成目标文本的语音（语音克隆/TTS）

**参数**：

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `audio_path` | string | ✅ | - | 参考音频文件路径 |
| `ref_text` | string | ✅ | - | 参考音频的文本内容 |
| `target_text` | string | ✅ | - | 要生成的文本内容 |
| `output_path` | string | ❌ | 自动生成 | 输出文件路径 |
| `system_prompt` | string | ❌ | "Convert the text to speech" | 系统提示词 |
| `sample_rate` | int | ❌ | 24000 | 采样率（16000, 22050, 24000, 28000, 32000） |
| `temperature` | float | ❌ | 1.0 | 温度参数（0.0-1.5） |
| `top_k` | int | ❌ | 50 | Top-K 采样（1-100） |
| `top_p` | float | ❌ | 0.9 | Nucleus 采样（0.0-1.0） |
| `penalty` | float | ❌ | 1.2 | 重复惩罚（0.0-2.0） |
| `random_seed` | int | ❌ | 49 | 随机种子 |

**返回值**：

```json
{
  "status": "success",
  "output_path": "/app/outputs/generated_1234567890.wav",
  "sample_rate": 24000,
  "duration_seconds": 5.2
}
```

**示例**：

```python
result = await session.call_tool("generate_speech", {
    "audio_path": "/app/temp/reference.wav",
    "ref_text": "你好，欢迎使用语音合成系统",
    "target_text": "今天天气很好，适合出门散步",
    "temperature": 1.0,
    "sample_rate": 24000
})

if result["status"] == "success":
    print(f"生成成功: {result['output_path']}")
    print(f"时长: {result['duration_seconds']} 秒")
```

### 2. transcribe_audio - 转录音频

**描述**：将音频文件转换为文字（ASR）

**参数**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `audio_path` | string | ✅ | 音频文件路径 |

**返回值**：

```json
{
  "status": "success",
  "text": "这是转录的文字内容",
  "language": "zh"
}
```

**示例**：

```python
result = await session.call_tool("transcribe_audio", {
    "audio_path": "/app/temp/audio.wav"
})

if result["status"] == "success":
    print(f"转录结果: {result['text']}")
    print(f"语言: {result['language']}")
```

### 3. get_gpu_status - 获取 GPU 状态

**描述**：查询当前 GPU 和模型状态

**参数**：无

**返回值**：

```json
{
  "status": "success",
  "data": {
    "Llasa-8B": {
      "model_name": "Llasa-8B",
      "location": "CPU",
      "idle_time_seconds": 123.4,
      "idle_timeout_seconds": 600,
      "monitor_running": true,
      "gpu_memory": {
        "GPU_0": {
          "allocated_gb": 0.5,
          "reserved_gb": 0.6,
          "total_gb": 24.0
        }
      }
    }
  }
}
```

**示例**：

```python
result = await session.call_tool("get_gpu_status", {})

if result["status"] == "success":
    for model_name, status in result["data"].items():
        print(f"{model_name}: {status['location']}")
        print(f"  空闲时间: {status['idle_time_seconds']} 秒")
```

### 4. offload_gpu - 卸载模型到 CPU

**描述**：手动将所有模型从 GPU 卸载到 CPU，释放 GPU 显存

**参数**：无

**返回值**：

```json
{
  "status": "success",
  "message": "所有模型已卸载到 CPU，GPU 显存已释放"
}
```

**示例**：

```python
result = await session.call_tool("offload_gpu", {})
print(result["message"])
```

### 5. release_gpu - 完全释放模型

**描述**：完全释放所有模型（清空 GPU 和 CPU 缓存）

**参数**：无

**返回值**：

```json
{
  "status": "success",
  "message": "所有模型已完全释放"
}
```

**示例**：

```python
result = await session.call_tool("release_gpu", {})
print(result["message"])
```

### 6. update_gpu_timeout - 更新超时时间

**描述**：更新 GPU 空闲超时时间

**参数**：

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `timeout_seconds` | int | ✅ | 超时时间（秒） |

**返回值**：

```json
{
  "status": "success",
  "message": "GPU 空闲超时已更新为 300 秒"
}
```

**示例**：

```python
result = await session.call_tool("update_gpu_timeout", {
    "timeout_seconds": 300
})
print(result["message"])
```

## 📝 使用场景

### 场景 1: 批量语音生成

```python
import asyncio
from mcp import ClientSession
from pathlib import Path

async def batch_generate(texts, reference_audio, reference_text):
    """批量生成语音"""
    async with ClientSession() as session:
        tasks = []
        for i, text in enumerate(texts):
            task = session.call_tool("generate_speech", {
                "audio_path": reference_audio,
                "ref_text": reference_text,
                "target_text": text,
                "output_path": f"/app/outputs/batch_{i}.wav"
            })
            tasks.append(task)

        # 并发执行（注意：实际会受 GPU 限制）
        results = await asyncio.gather(*tasks)

        return results

# 使用
texts = [
    "第一段要生成的文字",
    "第二段要生成的文字",
    "第三段要生成的文字"
]

results = asyncio.run(batch_generate(
    texts=texts,
    reference_audio="/app/temp/reference.wav",
    reference_text="参考音频的文字"
))

for i, result in enumerate(results):
    if result["status"] == "success":
        print(f"✅ 第 {i+1} 段生成成功: {result['output_path']}")
    else:
        print(f"❌ 第 {i+1} 段生成失败: {result['error']}")
```

### 场景 2: 自动转录 + 生成

```python
async def transcribe_and_generate(audio_file, target_text):
    """先转录参考音频，再生成目标语音"""
    async with ClientSession() as session:
        # 1. 转录参考音频
        transcribe_result = await session.call_tool("transcribe_audio", {
            "audio_path": audio_file
        })

        if transcribe_result["status"] != "success":
            return {"error": "转录失败"}

        ref_text = transcribe_result["text"]
        print(f"转录结果: {ref_text}")

        # 2. 生成目标语音
        generate_result = await session.call_tool("generate_speech", {
            "audio_path": audio_file,
            "ref_text": ref_text,
            "target_text": target_text
        })

        return generate_result

# 使用
result = asyncio.run(transcribe_and_generate(
    audio_file="/app/temp/my_voice.wav",
    target_text="这是我想说的话"
))

if result["status"] == "success":
    print(f"生成成功: {result['output_path']}")
```

### 场景 3: 智能资源管理

```python
async def smart_batch_processing(audio_files):
    """智能批量处理，自动管理 GPU 资源"""
    async with ClientSession() as session:
        results = []

        for i, audio_file in enumerate(audio_files):
            # 1. 处理文件
            result = await session.call_tool("transcribe_audio", {
                "audio_path": audio_file
            })
            results.append(result)

            # 2. 每处理 10 个文件，手动卸载一次
            if (i + 1) % 10 == 0:
                await session.call_tool("offload_gpu", {})
                print(f"已处理 {i+1} 个文件，GPU 已卸载")

        # 3. 全部完成后，完全释放
        await session.call_tool("release_gpu", {})
        print("全部处理完成，GPU 已释放")

        return results
```

### 场景 4: 监控和告警

```python
async def monitor_gpu():
    """监控 GPU 状态，超过阈值时告警"""
    async with ClientSession() as session:
        while True:
            # 获取 GPU 状态
            result = await session.call_tool("get_gpu_status", {})

            if result["status"] == "success":
                data = result["data"]

                for model_name, status in data.items():
                    gpu_mem = status["gpu_memory"].get("GPU_0", {})
                    allocated = gpu_mem.get("allocated_gb", 0)
                    total = gpu_mem.get("total_gb", 24)

                    # 显存使用率超过 90%
                    usage_ratio = allocated / total
                    if usage_ratio > 0.9:
                        print(f"⚠️  警告: {model_name} 显存使用率 {usage_ratio*100:.1f}%")

                        # 尝试卸载
                        await session.call_tool("offload_gpu", {})
                        print("已尝试卸载模型")

            # 每 30 秒检查一次
            await asyncio.sleep(30)
```

## 🔧 高级配置

### 自定义 MCP 服务器位置

如果您在非 Docker 环境中运行：

```json
{
  "mcpServers": {
    "llasa-tts": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "GPU_IDLE_TIMEOUT": "600",
        "LLASA_MODEL_PATH": "/path/to/models/Llasa-8B",
        "XCODEC_MODEL_PATH": "/path/to/models/xcodec2"
      }
    }
  }
}
```

### 使用远程 MCP 服务器

通过 SSH 连接远程服务器：

```json
{
  "mcpServers": {
    "llasa-tts-remote": {
      "command": "ssh",
      "args": [
        "user@remote-server",
        "docker exec -i llasa-tts-8b-webui python /app/mcp_server.py"
      ]
    }
  }
}
```

## 🐛 故障排除

### 问题 1: MCP 服务器无法连接

**症状**：客户端连接超时

**解决方案**：

```bash
# 1. 检查容器状态
docker ps | grep llasa

# 2. 检查 MCP 服务器是否运行
docker exec llasa-tts-8b-webui ps aux | grep mcp

# 3. 手动测试 MCP 服务器
docker exec -it llasa-tts-8b-webui python /app/mcp_server.py

# 4. 查看日志
docker logs llasa-tts-8b-webui | grep MCP
```

### 问题 2: 文件路径错误

**症状**：`audio_path` 找不到文件

**原因**：MCP 在容器内运行，需要使用容器内的路径

**解决方案**：

```python
# ❌ 错误：使用宿主机路径
"audio_path": "/home/user/audio.wav"

# ✅ 正确：使用容器内路径
"audio_path": "/app/temp/audio.wav"

# 或者先复制文件到挂载目录
# 宿主机: ./temp/audio.wav
# 容器内: /app/temp/audio.wav
```

### 问题 3: 工具调用超时

**症状**：长时间无响应

**原因**：首次加载模型需要 20-30 秒

**解决方案**：

```python
# 增加超时时间
async with ClientSession(timeout=120) as session:  # 120 秒超时
    result = await session.call_tool("generate_speech", {...})
```

## 📚 相关文档

- [GPU 管理文档](./GPU_MANAGEMENT.md) - GPU 智能管理详解
- [Docker 部署指南](./DOCKER_GUIDE.md) - Docker 部署说明
- [API 文档](http://localhost:7861/apidocs) - REST API 接口文档

## 🤝 支持

如有问题，请查看：
1. [项目 README](./README.md)
2. [常见问题](./FAQ.md)
3. [GitHub Issues](https://github.com/your-repo/issues)

## 📖 参考资料

- [FastMCP 官方文档](https://github.com/jlowin/fastmcp)
- [Model Context Protocol 规范](https://modelcontextprotocol.io/)
- [Llasa-8B 模型](https://huggingface.co/HKUSTAudio/Llasa-8B)
