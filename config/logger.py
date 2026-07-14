"""
统一日志模块

功能：
- 同时输出到控制台（Streamlit 中可见）和文件
- 按日期自动轮转，保留最近 7 天日志
- 全局单例 logger，各模块通过 get_logger(__name__) 获取
"""
import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

_loggers: dict = {}
_initialized = False

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _ensure_log_dir() -> None:
    """确保日志目录存在。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _init_root_logger() -> None:
    """初始化根 logger，配置 handlers。"""
    global _initialized
    if _initialized:
        return

    _ensure_log_dir()

    root = logging.getLogger("jobsearch_agent")
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # 格式
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler（Streamlit 运行时可见于终端）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # 文件 handler（按天轮转，保留 7 天）
    try:
        file_handler = TimedRotatingFileHandler(
            str(LOG_FILE),
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        root.info(f"日志系统初始化完成 | 级别={LOG_LEVEL} | 文件={LOG_FILE}")
    except Exception:
        # 文件日志初始化失败不阻塞应用，仅输出警告
        root.warning(f"无法创建日志文件 {LOG_FILE}，仅输出到控制台")

    _initialized = True


def get_logger(name: str = "jobsearch_agent") -> logging.Logger:
    """
    获取指定名称的 logger 实例（按 name 缓存）。
    
    使用方式：
        from config.logger import get_logger
        logger = get_logger(__name__)
        logger.info("正在加载知识库...")
        logger.error("加载失败", exc_info=True)
    
    Args:
        name: logger 名称，建议使用 __name__
    
    Returns:
        logging.Logger 实例
    """
    if not _initialized:
        _init_root_logger()

    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)

    return _loggers[name]


def get_log_file_path() -> Optional[Path]:
    """返回当前日志文件路径（供 UI 展示下载链接）。"""
    if LOG_FILE.exists():
        return LOG_FILE
    return None