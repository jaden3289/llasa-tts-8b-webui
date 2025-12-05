# GPU 智能显存管理系统

## 🎯 核心特性

本项目实现了**智能 GPU 显存管理系统**，实现：

1. **懒加载（Lazy Loading）**：首次请求时才加载模型到 GPU
2. **即用即卸（Immediate Offload）**：任务完成后立即卸载到 CPU
3. **自动监控（Auto Monitor）**：空闲超时后自动释放资源

## 📊 状态转换图

```
未加载 ──首次请求(20-30s)──→ GPU ──任务完成(2s)──→ CPU ──新请求(2-5s)──→ GPU
  ↑                                                    ↓
  └──────────────超时/手动释放(1s)─────────────────────┘
```

## 🔧 工作原理

### 1. 懒加载（Lazy Loading）

模型**不会**在启动时立即加载到 GPU，而是在**首次使用时**才加载：

```python
# 首次调用（加载到 GPU，20-30秒）
model = gpu_manager.get_model(load_func=load_model_function, model_name="MyModel")

# 模型现在在 GPU 上
result = model.process(data)
```

**好处**：
- 启动速度快
- 不占用不必要的显存
- 支持多个模型按需加载

### 2. 即用即卸（Immediate Offload）

任务完成后**立即**将模型从 GPU 转移到 CPU，释放显存：

```python
# 处理任务
result = model.process(data)

# 立即卸载（2秒）
gpu_manager.force_offload()

# 现在 GPU 显存已释放，模型在 CPU 缓存中
```

**好处**：
- GPU 显存使用量降到最低（< 1GB）
- 其他 GPU 容器可以使用更多显存
- 支持多个服务共享 GPU

### 3. 快速恢复

当再次需要模型时，从 CPU 快速转移回 GPU（2-5秒）：

```python
# 第二次调用（从 CPU 恢复，2-5秒）
model = gpu_manager.get_model(load_func=load_model_function, model_name="MyModel")

# 比首次加载快 5-10 倍！
```

### 4. 自动监控

后台监控线程定期检查模型空闲时间：

- 如果模型在 GPU 上空闲超过设定时间（默认 600 秒）
- 自动将模型转移到 CPU
- 如果长时间不用，可完全释放

## 📈 性能对比

| 操作 | 时间 | 显存占用 |
|------|------|----------|
| 首次加载（从磁盘） | 20-30 秒 | 24 GB |
| 处理任务 | 视任务而定 | 24 GB |
| 卸载到 CPU | 2 秒 | < 1 GB |
| 从 CPU 恢复到 GPU | 2-5 秒 | 24 GB |
| 完全释放 | 1 秒 | 0 GB |

## 🛠️ 使用方法

### 方式一：通过 API

```bash
# 获取 GPU 状态
curl http://localhost:7861/api/gpu/status

# 手动卸载到 CPU
curl -X POST http://localhost:7861/api/gpu/offload

# 完全释放
curl -X POST http://localhost:7861/api/gpu/release
```

### 方式二：通过 MCP

```python
from mcp import ClientSession

async with ClientSession() as session:
    # 获取 GPU 状态
    status = await session.call_tool("get_gpu_status", {})

    # 手动卸载
    result = await session.call_tool("offload_gpu", {})

    # 完全释放
    result = await session.call_tool("release_gpu", {})
```

### 方式三：在代码中使用

```python
from gpu_manager import get_global_manager

# 获取全局管理器
gpu_manager = get_global_manager(idle_timeout=600)

# 注册模型
llasa_manager = gpu_manager.register_model("Llasa-8B")

# 定义加载函数
def load_llasa():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        'HKUSTAudio/Llasa-8B',
        torch_dtype=torch.float16,
        device_map='cuda'
    )
    return model

# 使用模型（标准 3 步流程）
try:
    # 步骤 1: 懒加载
    model = llasa_manager.get_model(load_func=load_llasa, model_name="Llasa-8B")

    # 步骤 2: 处理
    result = model.generate(...)

    # 步骤 3: 立即卸载（关键！）
    llasa_manager.force_offload()

    return result

except Exception as e:
    # 异常时也要卸载
    llasa_manager.force_offload()
    raise e
```

## ⚙️ 配置参数

### 环境变量

在 `.env` 文件中配置：

```bash
# GPU 空闲超时时间（秒）
# 模型在 GPU 上空闲超过此时间后会自动卸载到 CPU
GPU_IDLE_TIMEOUT=600

# GPU ID（由 start.sh 自动选择最空闲的 GPU）
NVIDIA_VISIBLE_DEVICES=0
```

### 运行时调整

```bash
# API 方式（需要实现 API 端点）
curl -X POST http://localhost:7861/api/gpu/timeout \
  -H "Content-Type: application/json" \
  -d '{"timeout": 300}'

# MCP 方式
mcp call update_gpu_timeout '{"timeout_seconds": 300}'
```

### 代码方式

```python
# 更新超时时间
gpu_manager.update_all_timeout(new_timeout=300)

# 查看当前状态
status = gpu_manager.get_all_status()
print(status)
```

## 🔍 监控和调试

### 查看日志

```bash
# Docker 日志
docker-compose logs -f llasa-tts-webui

# 查找 GPU 相关日志
docker-compose logs llasa-tts-webui | grep "GPU\|显存\|卸载\|加载"
```

### 监控 GPU 使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 只看显存
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# 进入容器内部查看
docker exec -it llasa-tts-8b-webui bash
nvidia-smi
```

### 状态 API

```bash
# 获取详细状态
curl http://localhost:7861/api/gpu/status | jq

# 示例输出
{
  "Llasa-8B": {
    "model_name": "Llasa-8B",
    "location": "CPU",
    "idle_time_seconds": 45.2,
    "idle_timeout_seconds": 600,
    "monitor_running": true,
    "gpu_memory": {
      "GPU_0": {
        "allocated_gb": 0.5,
        "reserved_gb": 0.6,
        "total_gb": 24.0
      }
    }
  },
  "XCodec2": {
    "model_name": "XCodec2",
    "location": "未加载",
    ...
  }
}
```

## 💡 最佳实践

### 1. 确保异常安全

```python
try:
    model = manager.get_model(load_func, "ModelName")
    result = model.process(data)
    manager.force_offload()  # 正常完成后卸载
    return result
except Exception as e:
    manager.force_offload()  # 异常时也要卸载
    raise e
```

### 2. 多模型协调

```python
# 注册多个模型
codec_manager = gpu_manager.register_model("XCodec2")
llasa_manager = gpu_manager.register_model("Llasa-8B")

# 交替使用，避免同时占用
codec_model = codec_manager.get_model(load_codec, "XCodec2")
encoded = codec_model.encode(data)
codec_manager.force_offload()  # 用完就卸载

llasa_model = llasa_manager.get_model(load_llasa, "Llasa-8B")
generated = llasa_model.generate(encoded)
llasa_manager.force_offload()  # 用完就卸载

# 再次需要 codec 时，快速恢复
codec_model = codec_manager.get_model(load_codec, "XCodec2")
decoded = codec_model.decode(generated)
codec_manager.force_offload()
```

### 3. 调整超时时间

根据使用场景调整：

- **开发/测试环境**：设置较短超时（60-300 秒），快速释放资源
- **生产环境（频繁使用）**：设置较长超时（600-1800 秒），避免频繁重载
- **生产环境（偶尔使用）**：设置较短超时（60-180 秒），节省资源

### 4. 与其他 GPU 容器共存

```bash
# 启动时自动选择最空闲的 GPU
./start.sh

# 或手动指定 GPU
NVIDIA_VISIBLE_DEVICES=1 docker-compose up -d
```

## 🐛 故障排除

### 问题 1: 显存没有释放

**症状**：调用 `force_offload()` 后显存仍然很高

**原因**：
- Python 垃圾回收延迟
- 还有其他引用指向模型

**解决方案**：
```python
import gc
import torch

manager.force_offload()
gc.collect()  # 强制垃圾回收
torch.cuda.empty_cache()  # 清空 CUDA 缓存
```

### 问题 2: 模型加载失败

**症状**：`get_model()` 抛出异常

**原因**：
- 显存不足
- 模型路径错误
- 网络问题（下载模型失败）

**解决方案**：
```bash
# 1. 检查显存
nvidia-smi

# 2. 检查模型路径
ls -la models_cache/

# 3. 预先下载模型
huggingface-cli download HKUSTAudio/Llasa-8B

# 4. 使用本地模型
export LLASA_MODEL_PATH=/path/to/local/model
```

### 问题 3: 自动监控不工作

**症状**：超时后模型没有自动卸载

**原因**：
- 监控线程未启动
- 超时时间设置过长

**解决方案**：
```python
# 确保启动监控
gpu_manager = get_global_manager(idle_timeout=600)
# 监控线程应该自动启动

# 检查监控状态
status = gpu_manager.get_all_status()
print(status['monitor_running'])  # 应该是 True

# 手动启动（如果需要）
for manager in gpu_manager.managers.values():
    manager.start_monitor()
```

## 📚 相关文档

- [MCP 使用指南](./MCP_GUIDE.md) - MCP 接口使用方法
- [Docker 部署指南](./DOCKER_GUIDE.md) - Docker 部署详细说明
- [快速开始](./QUICK_START.md) - 快速上手指南

## 🤝 贡献

如果您在使用中遇到问题或有改进建议，欢迎提交 Issue 或 Pull Request！
