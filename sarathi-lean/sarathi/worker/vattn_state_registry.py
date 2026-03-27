# sarathi/worker/vattn_state_registry.py
"""
Tiny module-level registry that holds a reference to the active
vATTNCacheEngine so ModelRunner can query the Mamba slot map
without creating a circular import.

  base_worker  →  vattn_state_registry  ←  model_runner
                        ↑
               (no back-edge to base_worker)
"""
from typing import Optional

_cache_engine = None


def register_cache_engine(engine) -> None:
    global _cache_engine
    _cache_engine = engine


def get_vattn_slot_map() -> Optional[dict]:
    """Return {seq_id: slot_id} for every active request, or None."""
    if _cache_engine is None:
        return None
    if not hasattr(_cache_engine, "get_mamba_slot_map"):
        return None
    return _cache_engine.get_mamba_slot_map()
    