#!/bin/bash

# =============================================================================
# Llasa-8B TTS API 测试脚本
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# 配置
API_HOST=${API_HOST:-localhost}
API_PORT=${API_PORT:-7861}
BASE_URL="http://${API_HOST}:${API_PORT}"

print_header "🧪 Llasa-8B TTS API 测试"

# =============================================================================
# 1. 健康检查
# =============================================================================

print_info "测试 1: 健康检查"
response=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health")

if [ "$response" -eq 200 ]; then
    print_success "健康检查通过 (HTTP 200)"
else
    print_error "健康检查失败 (HTTP $response)"
    print_error "请确保服务已启动：docker-compose ps"
    exit 1
fi

# =============================================================================
# 2. GPU 状态查询
# =============================================================================

print_info "测试 2: 查询 GPU 状态"
response=$(curl -s "${BASE_URL}/api/gpu/status")

if echo "$response" | jq . >/dev/null 2>&1; then
    print_success "GPU 状态查询成功"
    echo "$response" | jq .
else
    print_error "GPU 状态查询失败"
    echo "$response"
fi

# =============================================================================
# 3. API 文档测试
# =============================================================================

print_info "测试 3: 检查 API 文档"
response=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/apidocs")

if [ "$response" -eq 200 ]; then
    print_success "API 文档可访问 (${BASE_URL}/apidocs)"
else
    print_error "API 文档不可访问 (HTTP $response)"
fi

# =============================================================================
# 4. 转录测试（需要音频文件）
# =============================================================================

if [ -f "test_audio.wav" ]; then
    print_info "测试 4: 音频转录"

    response=$(curl -s -X POST "${BASE_URL}/api/transcribe" \
        -F "audio=@test_audio.wav")

    if echo "$response" | jq -e '.text' >/dev/null 2>&1; then
        print_success "转录成功"
        echo "转录结果："
        echo "$response" | jq '.text'
    else
        print_error "转录失败"
        echo "$response"
    fi
else
    print_info "测试 4: 跳过（未找到 test_audio.wav）"
    echo "提示：将测试音频命名为 test_audio.wav 可测试转录功能"
fi

# =============================================================================
# 5. GPU 管理测试
# =============================================================================

print_info "测试 5: GPU 卸载功能"
response=$(curl -s -X POST "${BASE_URL}/api/gpu/offload")

if echo "$response" | jq -e '.status == "success"' >/dev/null 2>&1; then
    print_success "GPU 卸载成功"
    echo "$response" | jq '.message'
else
    print_error "GPU 卸载失败"
    echo "$response"
fi

# =============================================================================
# 6. 端点列表
# =============================================================================

print_header "📋 可用端点"

cat <<EOF
Web UI:
  ${BASE_URL/:7861/:7860}/

API 端点:
  GET  ${BASE_URL}/health
  GET  ${BASE_URL}/api/gpu/status
  POST ${BASE_URL}/api/gpu/offload
  POST ${BASE_URL}/api/gpu/release
  POST ${BASE_URL}/api/transcribe
  POST ${BASE_URL}/api/tts

API 文档 (Swagger):
  ${BASE_URL}/apidocs
EOF

# =============================================================================
# 7. 性能测试（可选）
# =============================================================================

if command -v ab &> /dev/null && [ "$1" == "--benchmark" ]; then
    print_header "⚡ 性能测试"

    print_info "运行 100 个请求测试健康检查端点"
    ab -n 100 -c 10 "${BASE_URL}/health"

    print_info "运行 10 个请求测试 GPU 状态端点"
    ab -n 10 -c 2 "${BASE_URL}/api/gpu/status"
fi

# =============================================================================
# 总结
# =============================================================================

print_header "✅ 测试完成"

echo "提示："
echo "  • 使用 --benchmark 参数运行性能测试"
echo "  • 查看完整 API 文档：${BASE_URL}/apidocs"
echo "  • 查看日志：docker-compose logs -f"
echo ""
