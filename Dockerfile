# =============================================================================
# Llasa-TTS-8B Docker Image - 智能 GPU 管理版
# 使用 PyTorch 官方的 CUDA 镜像（已经配置好 GPU 支持）
# =============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量（移除硬编码的 CUDA_VISIBLE_DEVICES，由 start.sh 动态设置）
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_ENDPOINT=https://hf-mirror.com \
    GPU_IDLE_TIMEOUT=600

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    wget \
    sox \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 xcodec2（正常安装，让 pip 自动处理依赖）
RUN pip install --no-cache-dir xcodec2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 重新安装 torch 2.6.0 和兼容的 torchao（覆盖 xcodec2 降级的版本）
RUN pip install --no-cache-dir --force-reinstall torch==2.6.0 torchaudio==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir --upgrade torchao -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 API 和 MCP 相关依赖 (fixed fastmcp version issue)
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    flasgger==0.9.7.1 \
    pydantic>=2.5.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir fastmcp

# 复制应用代码
COPY gpu_manager.py .
COPY app.py .
COPY api_server.py .
COPY mcp_server.py .
COPY main.py .

# 赋予执行权限
RUN chmod +x main.py

# 创建必要目录
RUN mkdir -p /root/.cache/huggingface /app/outputs /app/temp

# 暴露端口
# 7860: Gradio Web UI
# 7861: REST API
# 7862: MCP Server (if standalone)
EXPOSE 7860 7861 7862

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# 启动命令（统一启动脚本：UI + API + MCP）
CMD ["python", "main.py"]
