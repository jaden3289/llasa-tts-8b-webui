#!/bin/bash

# =============================================================================
# 项目完整性验证脚本
# 检查所有必需文件和配置
# =============================================================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Llasa-TTS-8B 项目完整性验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ERRORS=0

# 核心文件检查
echo "📁 核心文件检查："
FILES=(
    "main.py"
    "app.py"
    "api_server.py"
    "mcp_server.py"
    "gpu_manager.py"
    "Dockerfile"
    "docker-compose.yml"
    "requirements.txt"
    ".env.example"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "$file"
    else
        print_error "$file 缺失"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# 脚本文件检查
echo "📜 脚本文件检查："
SCRIPTS=(
    "start.sh"
    "stop.sh"
    "test_all.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            print_success "$script (可执行)"
        else
            print_warning "$script (不可执行，运行: chmod +x $script)"
        fi
    else
        print_error "$script 缺失"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# 文档文件检查
echo "📚 文档文件检查："
DOCS=(
    "README.md"
    "ARCHITECTURE.md"
    "GPU_MANAGEMENT.md"
    "MCP_GUIDE.md"
    "DOCKER_GUIDE.md"
    "QUICK_START.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        print_success "$doc"
    else
        print_warning "$doc 缺失（可选）"
    fi
done

echo ""

# 目录检查
echo "📂 目录检查："
DIRS=(
    "models_cache"
    "outputs"
    "temp"
)

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "$dir/"
    else
        print_warning "$dir/ 不存在（启动时会自动创建）"
    fi
done

echo ""

# Python 语法检查
echo "🐍 Python 语法检查："
PYTHON_FILES=(
    "main.py"
    "app.py"
    "api_server.py"
    "mcp_server.py"
    "gpu_manager.py"
)

for pyfile in "${PYTHON_FILES[@]}"; do
    if [ -f "$pyfile" ]; then
        if python3 -m py_compile "$pyfile" 2>/dev/null; then
            print_success "$pyfile 语法正确"
        else
            print_error "$pyfile 语法错误"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

echo ""

# Docker 环境检查
echo "🐳 Docker 环境检查："

if command -v docker &> /dev/null; then
    print_success "Docker 已安装"
else
    print_error "Docker 未安装"
    ERRORS=$((ERRORS + 1))
fi

if command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1; then
    print_success "Docker Compose 已安装"
else
    print_error "Docker Compose 未安装"
    ERRORS=$((ERRORS + 1))
fi

if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA 驱动已安装"
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "   检测到 ${GPU_COUNT} 个 GPU"
else
    print_error "NVIDIA 驱动未安装"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# .env 文件检查
echo "⚙️  配置文件检查："

if [ -f ".env" ]; then
    print_success ".env 文件存在"
    
    # 检查关键配置
    if grep -q "^NVIDIA_VISIBLE_DEVICES=" .env; then
        print_success "  GPU 配置已设置"
    else
        print_warning "  GPU 配置未设置"
    fi
    
    if grep -q "^UI_PORT=" .env; then
        print_success "  端口配置已设置"
    else
        print_warning "  端口配置未设置"
    fi
else
    print_warning ".env 文件不存在（启动时会自动创建）"
fi

echo ""

# 总结
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ 验证通过！项目已准备就绪${NC}"
    echo ""
    echo "下一步："
    echo "  1. 运行 ./start.sh 启动服务"
    echo "  2. 访问 http://localhost:7860 使用 Web UI"
    echo "  3. 访问 http://localhost:7861/apidocs 查看 API 文档"
else
    echo -e "${RED}❌ 发现 ${ERRORS} 个错误，请修复后再启动${NC}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

exit $ERRORS
