"""
KV Cache 分配过程日志工具
========================
将 hybrid KV cache 的分配、释放、缓冲区规划等信息写入独立日志文件。

用法：
    from sarathi.core.kv_cache_logger import kv_logger
    kv_logger.info("消息")
    kv_logger.block_alloc("消息")
"""

import logging
import os
from datetime import datetime


def _build_logger(name: str, log_dir: str, filename: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler（多次 import 时）
    if logger.handlers:
        return logger

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 不向上传播，避免混入 sarathi 主日志
    logger.propagate = False

    logger.info(f"KV Cache 日志初始化，路径: {log_path}")
    return logger


# ── 三个独立日志文件，分别记录不同粒度的信息 ──

LOG_DIR = os.environ.get("KV_CACHE_LOG_DIR", "./kv_cache_logs")
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. 缓冲区规划（spec / group / GPU buffer 形状）—— 只在初始化时写一次
_layout_logger  = _build_logger(
    "kv_cache.layout",
    LOG_DIR,
    f"kv_layout_{_ts}.log",
)

# 2. Block 分配/释放（每次 allocate / free / remove_skipped）
_alloc_logger   = _build_logger(
    "kv_cache.alloc",
    LOG_DIR,
    f"kv_alloc_{_ts}.log",
)

# 3. BlockPool 底层（get_new_blocks / free_blocks，最细粒度）
_pool_logger    = _build_logger(
    "kv_cache.pool",
    LOG_DIR,
    f"kv_pool_{_ts}.log",
)


class KVCacheLogger:
    """统一入口，按类别路由到不同日志文件。"""

    # ── 缓冲区规划 ──
    def layout(self, msg: str) -> None:
        _layout_logger.info(msg)

    def layout_debug(self, msg: str) -> None:
        _layout_logger.debug(msg)

    # ── Block 分配/释放 ──
    def alloc(self, msg: str) -> None:
        _alloc_logger.info(msg)

    def alloc_debug(self, msg: str) -> None:
        _alloc_logger.debug(msg)

    # ── BlockPool 底层 ──
    def pool(self, msg: str) -> None:
        _pool_logger.info(msg)

    def pool_debug(self, msg: str) -> None:
        _pool_logger.debug(msg)


# 全局单例，其他模块直接 import 使用
kv_logger = KVCacheLogger()

