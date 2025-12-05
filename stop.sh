#!/bin/bash

# =============================================================================
# Llasa-8B TTS WebUI 停止脚本
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo ""
echo "=========================================="
echo "🛑 停止 Llasa-8B TTS WebUI"
echo "=========================================="
echo ""

# 检查 Docker Compose 命令
DOCKER_COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE_CMD="docker compose"
else
    print_error "Docker Compose 未找到"
    exit 1
fi

# 显示容器状态
print_info "当前容器状态:"
$DOCKER_COMPOSE_CMD ps

echo ""
print_info "正在停止并删除容器..."

# 停止容器
$DOCKER_COMPOSE_CMD down

if [ $? -eq 0 ]; then
    print_success "服务已成功停止"
    echo ""
    echo "注意："
    echo "  • 模型缓存已保留在 ./models_cache 目录"
    echo "  • 输出文件已保留在 ./outputs 目录"
    echo "  • 若要完全清理，请手动删除这些目录"
    echo ""
else
    print_error "停止失败"
    exit 1
fi
