#!/usr/bin/env python3
"""
Llasa-TTS-8B 统一启动脚本
同时启动：Web UI + REST API + MCP Server
"""

import os
import sys
import logging
import threading
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# 端口配置
UI_PORT = int(os.getenv('UI_PORT', 7860))
API_PORT = int(os.getenv('API_PORT', 7861))
MCP_PORT = int(os.getenv('MCP_PORT', 7862))

def start_gradio_ui():
    """启动 Gradio Web UI"""
    logger.info(f"🎨 启动 Gradio Web UI (端口 {UI_PORT})...")
    
    # 导入并运行 app.py
    import app
    app.app.launch(
        server_name="0.0.0.0",
        server_port=UI_PORT,
        share=False,
        show_error=True
    )

def start_api_server():
    """启动 REST API 服务器"""
    logger.info(f"🔌 启动 REST API 服务器 (端口 {API_PORT})...")
    
    # 等待一下让 UI 先启动
    time.sleep(2)
    
    import api_server
    api_server.app.run(
        host='0.0.0.0',
        port=API_PORT,
        debug=False,
        threaded=True
    )

def start_mcp_server():
    """启动 MCP 服务器"""
    logger.info(f"🔗 MCP 服务器已集成到 API 中")
    # MCP 通过 stdio 运行，不需要单独启动
    # 用户可以通过 `python mcp_server.py` 单独运行

def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🚀 Llasa-TTS-8B 统一启动")
    logger.info("=" * 60)
    logger.info(f"📊 Web UI:   http://0.0.0.0:{UI_PORT}")
    logger.info(f"📊 API:      http://0.0.0.0:{API_PORT}")
    logger.info(f"📊 API Doc:  http://0.0.0.0:{API_PORT}/apidocs")
    logger.info(f"📊 MCP:      python mcp_server.py (单独运行)")
    logger.info("=" * 60)
    
    # 创建必要目录
    Path('./outputs').mkdir(exist_ok=True)
    Path('./temp').mkdir(exist_ok=True)
    
    # 启动 API 服务器（后台线程）
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # 启动 Gradio UI（主线程）
    try:
        start_gradio_ui()
    except KeyboardInterrupt:
        logger.info("\n👋 正在关闭服务...")
        sys.exit(0)

if __name__ == "__main__":
    main()
