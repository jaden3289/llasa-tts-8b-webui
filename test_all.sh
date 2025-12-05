#!/bin/bash

# =============================================================================
# Llasa-TTS-8B 完整测试脚本
# 测试：Docker 部署 + GPU 管理 + UI + API + MCP
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_header() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n${GREEN}$1${NC}\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

# 读取端口配置
UI_PORT=$(grep "^UI_PORT=" .env 2>/dev/null | cut -d'=' -f2 || echo "7860")
API_PORT=$(grep "^API_PORT=" .env 2>/dev/null | cut -d'=' -f2 || echo "7861")

print_header "🧪 Llasa-TTS-8B 完整测试"

# =============================================================================
# 1. Docker 环境测试
# =============================================================================

print_header "1️⃣  Docker 环境测试"

print_info "检查容器状态..."
if docker ps | grep -q "llasa-tts-8b-webui"; then
    print_success "容器正在运行"
    docker ps | grep "llasa-tts-8b-webui"
else
    print_error "容器未运行，请先执行 ./start.sh"
    exit 1
fi

print_info "检查容器日志..."
docker logs llasa-tts-8b-webui --tail 20

# =============================================================================
# 2. GPU 状态测试
# =============================================================================

print_header "2️⃣  GPU 状态测试"

print_info "检查 GPU 使用情况..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader | while IFS=',' read -r id name mem_used mem_total util; do
    echo "GPU ${id}: ${name}"
    echo "  显存: ${mem_used} / ${mem_total}"
    echo "  利用率: ${util}"
done

# =============================================================================
# 3. 服务健康检查
# =============================================================================

print_header "3️⃣  服务健康检查"

print_info "测试 Web UI (端口 ${UI_PORT})..."
if curl -s -f "http://localhost:${UI_PORT}/" > /dev/null; then
    print_success "Web UI 可访问"
else
    print_error "Web UI 无法访问"
fi

print_info "测试 API 健康检查 (端口 ${API_PORT})..."
HEALTH_RESPONSE=$(curl -s "http://localhost:${API_PORT}/health")
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_success "API 健康检查通过"
    echo "响应: $HEALTH_RESPONSE"
else
    print_error "API 健康检查失败"
fi

print_info "测试 Swagger 文档..."
if curl -s -f "http://localhost:${API_PORT}/apidocs" > /dev/null; then
    print_success "Swagger 文档可访问"
else
    print_error "Swagger 文档无法访问"
fi

# =============================================================================
# 4. GPU 管理 API 测试
# =============================================================================

print_header "4️⃣  GPU 管理 API 测试"

print_info "获取 GPU 状态..."
GPU_STATUS=$(curl -s "http://localhost:${API_PORT}/api/gpu/status")
echo "$GPU_STATUS" | python3 -m json.tool 2>/dev/null || echo "$GPU_STATUS"

print_info "测试手动卸载..."
OFFLOAD_RESPONSE=$(curl -s -X POST "http://localhost:${API_PORT}/api/gpu/offload")
if echo "$OFFLOAD_RESPONSE" | grep -q "success"; then
    print_success "手动卸载成功"
else
    print_error "手动卸载失败"
fi

sleep 2

print_info "再次获取 GPU 状态（验证卸载）..."
GPU_STATUS=$(curl -s "http://localhost:${API_PORT}/api/gpu/status")
echo "$GPU_STATUS" | python3 -m json.tool 2>/dev/null || echo "$GPU_STATUS"

# =============================================================================
# 5. API 功能测试（可选）
# =============================================================================

print_header "5️⃣  API 功能测试（需要测试音频）"

if [ -f "test_audio.wav" ]; then
    print_info "测试音频转录..."
    TRANSCRIBE_RESPONSE=$(curl -s -X POST "http://localhost:${API_PORT}/api/transcribe" \
        -F "audio=@test_audio.wav")
    
    if echo "$TRANSCRIBE_RESPONSE" | grep -q "text"; then
        print_success "音频转录成功"
        echo "$TRANSCRIBE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$TRANSCRIBE_RESPONSE"
    else
        print_error "音频转录失败"
    fi
else
    print_info "跳过 API 功能测试（未找到 test_audio.wav）"
    echo "提示：创建 test_audio.wav 文件以测试完整功能"
fi

# =============================================================================
# 6. MCP 测试（可选）
# =============================================================================

print_header "6️⃣  MCP 测试"

print_info "MCP 服务器需要单独运行："
echo "  docker exec -it llasa-tts-8b-webui python mcp_server.py"
echo ""
echo "或在宿主机运行："
echo "  python mcp_server.py"

# =============================================================================
# 7. 显存占用测试
# =============================================================================

print_header "7️⃣  显存占用测试"

print_info "当前显存占用："
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

print_info "等待 5 秒后再次检查..."
sleep 5

print_info "5 秒后显存占用："
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# =============================================================================
# 测试总结
# =============================================================================

print_header "✅ 测试完成"

echo "📊 访问地址："
echo "   • Web UI:  http://localhost:${UI_PORT}"
echo "   • API:     http://localhost:${API_PORT}"
echo "   • API Doc: http://localhost:${API_PORT}/apidocs"
echo ""
echo "📝 查看日志："
echo "   docker logs -f llasa-tts-8b-webui"
echo ""
echo "🛑 停止服务："
echo "   ./stop.sh"
echo ""
