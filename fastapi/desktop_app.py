"""
交通事故风险预测系统 - 桌面应用入口
整合Vue3前端和FastAPI后端为独立的桌面应用程序
"""

import logging
import os
import socket
import sys
import threading
import time
from typing import Optional

import uvicorn
from dotenv import load_dotenv


def get_resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

env_path = get_resource_path('.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"已加载环境变量: {env_path}")
else:
    load_dotenv()
    logger.warning(f"未找到.env文件: {env_path}")


def find_free_port(start_port: int = 8000, max_port: int = 9000) -> int:
    """
    查找可用端口
    
    Args:
        start_port: 起始端口
        max_port: 最大端口
        
    Returns:
        int: 可用端口号
    """
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"无法在端口范围 {start_port}-{max_port} 中找到可用端口")


def check_port_available(port: int) -> bool:
    """
    检查端口是否可用
    
    Args:
        port: 端口号
        
    Returns:
        bool: 是否可用
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False


def start_fastapi_server(port: int) -> None:
    try:
        from main import app

        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            reload=False,
        )
        
        # 创建并启动服务器
        server = uvicorn.Server(config)
        
        logger.info(f"FastAPI服务器启动在端口: {port}")
        server.run()
        
    except Exception as e:
        logger.error(f"FastAPI服务器启动失败: {str(e)}")
        raise


def wait_for_server_ready(port: int, timeout: int = 60) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('127.0.0.1', port))
                return True
        except (socket.error, OSError):
            pass
        elapsed = int(time.time() - start_time)
        if elapsed % 5 == 0 and elapsed > 0:
            logger.info(f"等待服务器启动中... 已等待 {elapsed} 秒")
        time.sleep(1)
    return False


def create_webview_window(port: int) -> None:
    """
    创建pywebview桌面窗口
    
    Args:
        port: FastAPI服务器端口
    """
    try:
        import webview
    except ImportError:
        logger.error("pywebview未安装，请运行: pip install pywebview==4.4.1")
        sys.exit(1)
    
    logger.info("等待FastAPI服务器启动...")
    if not wait_for_server_ready(port):
        logger.error("FastAPI服务器启动超时（60秒）")
        sys.exit(1)
    
    logger.info("服务器端口已就绪，等待HTTP服务初始化...")
    time.sleep(2)
    
    # 配置窗口
    url = f"http://127.0.0.1:{port}/"
    
    window = webview.create_window(
        title="交通事故风险预测系统",
        url=url,
        width=1200,
        height=800,
        resizable=True,
        min_size=(800, 600),
        text_select=True
    )
    
    logger.info(f"桌面窗口已打开，正在加载: {url}")
    
    try:
        # 启动webview
        webview.start()
    except Exception as e:
        logger.error(f"webview启动失败: {str(e)}")
        raise


def main():
    """
    主入口函数
    """
    try:
        logger.info("启动交通事故风险预测系统桌面应用")
        
        # 1. 查找可用端口
        try:
            port = find_free_port()
            logger.info(f"找到可用端口: {port}")
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
        
        # 2. 在后台线程启动FastAPI服务器
        server_thread = threading.Thread(
            target=start_fastapi_server,
            args=(port,),
            daemon=True,
            name="FastAPIServer"
        )
        
        logger.info("正在后台启动FastAPI服务器...")
        server_thread.start()
        
        # 3. 等待一小段时间确保服务器开始启动
        time.sleep(1)
        
        # 4. 在主线程创建桌面窗口
        create_webview_window(port)
        
    except KeyboardInterrupt:
        logger.info("用户中断应用程序")
    except Exception as e:
        logger.error(f"应用程序启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("应用程序已退出")


if __name__ == "__main__":
    main()