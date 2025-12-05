#!/bin/bash

# =============================================================================
# Llasa-8B TTS WebUI 智能启动脚本
# 功能：
#   1. 自动检测环境依赖
#   2. 自动选择显存占用最少的 GPU
#   3. 创建必要的目录
#   4. 构建并启动 Docker 容器
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印标题
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_header "🚀 Llasa-8B TTS WebUI 智能启动脚本"

# =============================================================================
# 1. 环境检查
# =============================================================================

print_info "检查运行环境..."

# 检查 Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker 未安装"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose
DOCKER_COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    print_error "Docker Compose 未安装"
    echo "请先安装 Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# 检查 nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi 未找到，请确保已安装 NVIDIA 驱动"
    exit 1
fi

# 检查 nvidia-docker
print_info "检查 NVIDIA Docker 运行时..."
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    print_error "NVIDIA Docker 运行时未正确配置"
    echo "请先安装 nvidia-docker: https://github.com/NVIDIA/nvidia-docker"
    exit 1
fi

print_success "Docker 环境检查通过"
echo ""

# =============================================================================
# 2. 自动选择最空闲的 GPU
# =============================================================================

print_info "检测可用 GPU..."

# 获取 GPU 信息
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
print_info "检测到 ${GPU_COUNT} 个 GPU"

# 显示所有 GPU 的状态
echo ""
echo "当前 GPU 状态："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader | while IFS=',' read -r id name mem_used mem_total util; do
    echo "GPU ${id}: ${name}"
    echo "  显存: ${mem_used} / ${mem_total}"
    echo "  利用率: ${util}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
done
echo ""

# 自动选择显存占用最少的 GPU
print_info "自动选择显存占用最少的 GPU..."
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
         sort -t',' -k2 -n | head -1 | cut -d',' -f1)

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_ID)
GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $GPU_ID)
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i $GPU_ID)

print_success "已选择 GPU ${GPU_ID}: ${GPU_NAME}"
print_info "显存使用: ${GPU_MEM_USED} / ${GPU_MEM_TOTAL}"
echo ""

# 导出环境变量
export NVIDIA_VISIBLE_DEVICES=$GPU_ID
export CUDA_VISIBLE_DEVICES=0  # 在容器内映射为 GPU 0

# =============================================================================
# 3. 创建 .env 文件
# =============================================================================

print_info "配置环境变量..."

# 如果 .env 不存在，从 .env.example 复制
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_info "从 .env.example 创建 .env 文件..."
        cp .env.example .env
    else
        print_warning ".env.example 不存在，创建默认 .env 文件..."
        cat > .env << EOF
NVIDIA_VISIBLE_DEVICES=${GPU_ID}
CUDA_VISIBLE_DEVICES=0
GPU_IDLE_TIMEOUT=600
UI_PORT=7860
API_PORT=7861
MCP_PORT=7862
HF_ENDPOINT=https://hf-mirror.com
MODELS_CACHE_DIR=./models_cache
OUTPUTS_DIR=./outputs
TEMP_DIR=./temp
MEM_LIMIT=32g
SHM_SIZE=8g
EOF
    fi
fi

# 更新 .env 中的 GPU 配置
if command -v sed &> /dev/null; then
    # 使用 sed 更新 GPU ID
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^NVIDIA_VISIBLE_DEVICES=.*/NVIDIA_VISIBLE_DEVICES=${GPU_ID}/" .env
    else
        # Linux
        sed -i "s/^NVIDIA_VISIBLE_DEVICES=.*/NVIDIA_VISIBLE_DEVICES=${GPU_ID}/" .env
    fi
    print_success "已更新 .env 文件中的 GPU 配置"
else
    print_warning "sed 命令不可用，请手动更新 .env 文件中的 NVIDIA_VISIBLE_DEVICES=${GPU_ID}"
fi

# =============================================================================
# 4. 创建必要的目录
# =============================================================================

print_info "创建必要的目录..."

mkdir -p ./models_cache
mkdir -p ./outputs
mkdir -p ./temp

print_success "目录创建完成"
echo ""

# =============================================================================
# 5. 构建 Docker 镜像
# =============================================================================

print_info "检查是否需要构建 Docker 镜像..."

# 检查镜像是否存在
if docker images | grep -q "llasa-tts-8b.*latest"; then
    print_info "Docker 镜像已存在"
    read -p "是否重新构建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "构建 Docker 镜像（这可能需要几分钟）..."
        $DOCKER_COMPOSE_CMD build
    fi
else
    print_info "构建 Docker 镜像（这可能需要几分钟）..."
    $DOCKER_COMPOSE_CMD build
fi

# =============================================================================
# 6. 启动容器
# =============================================================================

print_info "启动 Docker 容器..."
$DOCKER_COMPOSE_CMD up -d

if [ $? -eq 0 ]; then
    print_success "容器启动成功!"
else
    print_error "容器启动失败"
    exit 1
fi

# =============================================================================
# 7. 显示访问信息
# =============================================================================

# 读取端口配置
UI_PORT=$(grep "^UI_PORT=" .env 2>/dev/null | cut -d'=' -f2 || echo "7860")
API_PORT=$(grep "^API_PORT=" .env 2>/dev/null | cut -d'=' -f2 || echo "7861")

# 获取本机 IP
if command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")
else
    LOCAL_IP="localhost"
fi

print_header "✅ Llasa-8B TTS WebUI 已成功启动!"

echo "📊 服务访问地址："
echo "   • Web UI:  http://localhost:${UI_PORT}"
echo "   • Web UI:  http://${LOCAL_IP}:${UI_PORT} (局域网访问)"
echo "   • API:     http://localhost:${API_PORT}"
echo "   • API Doc: http://localhost:${API_PORT}/apidocs"
echo ""
echo "🎮 GPU 信息："
echo "   • 使用 GPU: ${GPU_ID} (${GPU_NAME})"
echo "   • 显存状态: ${GPU_MEM_USED} / ${GPU_MEM_TOTAL}"
echo ""
echo "📝 常用命令："
echo "   • 查看日志: ${DOCKER_COMPOSE_CMD} logs -f"
echo "   • 停止服务: ./stop.sh 或 ${DOCKER_COMPOSE_CMD} down"
echo "   • 重启服务: ${DOCKER_COMPOSE_CMD} restart"
echo "   • 查看状态: ${DOCKER_COMPOSE_CMD} ps"
echo ""
echo "⏳ 首次启动需要下载模型（约 20GB），请耐心等待..."
echo "   可以通过以下命令查看下载进度:"
echo "   ${DOCKER_COMPOSE_CMD} logs -f llasa-tts-webui"
echo ""

print_header "🎉 祝使用愉快！"
