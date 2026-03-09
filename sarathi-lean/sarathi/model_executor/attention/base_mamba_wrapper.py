"""
base_mamba_wrapper.py
=====================
Mamba 状态管理器的抽象基类，类比 BaseAttentionWrapper。

与 Attention 的对应关系：
  AttentionWrapper.forward(q, k, v, kv_cache)  →  (num_blocks, block_size, num_heads, head_dim)
  MambaWrapper.forward(x, params, mamba_cache) →  (num_blocks, d_inner, d_conv-1)
                                                    (num_blocks, d_inner, d_state)

状态存储约定：
  MambaManager 通过 get_num_skipped_tokens = N-1 保证只保留最后一个有效 block，
  因此每条序列的 conv_state / ssm_state 始终存储在其最后一个非 null block 的 slot 中：
    block_id = [b for b in block_table if b != 0][-1]

调用时序（与 FlashAttentionWrapper 完全对称）：
  begin_forward(seq_metadata_list)
      ↓
  forward(x, ..., conv_state_cache, ssm_state_cache)   # 每层调一次
      ↓
  end_forward()
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

from sarathi.config import ModelConfig, ParallelConfig
from sarathi.core.datatypes.sequence import SequenceMetadata


class BaseMambaWrapper(ABC):
    """Mamba 状态管理器抽象基类。"""

    _inst = None

    # ------------------------------------------------------------------
    # 单例
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def init(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        block_size: int,
        device: torch.device,
        mamba_group_idx: int = 1,
    ) -> None:
        """
        Args:
            model_config      : 模型配置，用于读取 d_state / d_conv 等参数
            parallel_config   : 并行配置
            block_size        : KV Cache block 大小（token 数）
            device            : GPU 设备
            mamba_group_idx   : seq_metadata.block_tables 中 Mamba group 的下标
                                （build_kv_cache_config 按 trans→swa→state 顺序排列 group，
                                 纯 trans+state 模型时 mamba_group_idx=1）
        """
        self.device = device
        self.block_size = block_size
        self.mamba_group_idx = mamba_group_idx

        self.d_state: int = model_config.get_d_state()

        hf = model_config.hf_config
        self.d_conv: int = int(
            getattr(hf, "mamba_d_conv", None) or getattr(hf, "d_conv", 4)
        )

        # begin_forward 阶段填充
        self.is_metadata_initialized: bool = False
        self.is_profiling_iteration:  bool = False

        # 每条序列的元数据（与 token 顺序一致）
        self.seq_is_prefill:    List[bool] = []
        self.seq_lens:          List[int]  = []
        self.seq_state_block_ids: List[int] = []   # 状态存储的 block_id
        self.seq_token_offsets: List[int]  = []    # 在 hidden_states 中的起始位置

    # ------------------------------------------------------------------
    # Forward 接口
    # ------------------------------------------------------------------

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        """
        解析 seq_metadata_list，提取每条序列的：
          - is_prefill      : prefill 还是 decode
          - seq_len         : 本次处理的 token 数
          - state_block_id  : Mamba 状态存储的 block_id（最后一个非 null block）
          - token_offset    : 在 hidden_states 中的起始下标
        """
        self.seq_is_prefill     = []
        self.seq_lens           = []
        self.seq_state_block_ids = []
        self.seq_token_offsets  = []
        self.is_metadata_initialized = True
        self.is_profiling_iteration  = False

        token_offset = 0

        for seq_meta in seq_metadata_list:
            # Profiling 阶段：block_tables 尚未初始化
            if seq_meta.block_tables is None:
                self.is_profiling_iteration = True
                return

            is_prefill = seq_meta.is_prompt

            if is_prefill:
                seq_len = seq_meta.seq.get_next_prompt_chunk_len(
                    seq_meta.prompt_chunk_len
                )
            else:
                seq_len = 1

            # ── 找到最后一个有效 block_id ──────────────────────────────
            # block_tables[mamba_group_idx] 是 Mamba group 的 block_id 列表
            # block_id = 0 是 null_block（MambaManager 会把窗口外的全部替换为 null）
            mamba_table = seq_meta.block_tables[self.mamba_group_idx]
            valid_ids   = [bid for bid in mamba_table if bid != 0]
            state_block_id = valid_ids[-1] if valid_ids else 0

            self.seq_is_prefill.append(is_prefill)
            self.seq_lens.append(seq_len)
            self.seq_state_block_ids.append(state_block_id)
            self.seq_token_offsets.append(token_offset)
            token_offset += seq_len

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        # ── Conv1d 参数 ──
        conv1d_weight: torch.Tensor,        # (d_inner, 1, d_conv)
        conv1d_bias:   Optional[torch.Tensor],  # (d_inner,) | None
        # ── SSM 参数 ──
        x_proj_weight: torch.Tensor,        # (dt_rank + 2*d_state, d_inner)
        dt_proj_weight: torch.Tensor,       # (d_inner, dt_rank)
        dt_proj_bias:  torch.Tensor,        # (d_inner,)
        A_log:         torch.Tensor,        # (d_inner, d_state)
        D:             torch.Tensor,        # (d_inner,)
        # ── Cache ──
        conv_state_cache: torch.Tensor,     # (num_blocks, d_inner, d_conv-1)
        ssm_state_cache:  torch.Tensor,     # (num_blocks, d_inner, d_state)
        # ── 其他 ──
        dt_softplus: bool = True,
        layer_id:    Optional[int] = None,
    ) -> torch.Tensor:
        """
        执行 Mamba forward，返回 output (total_tokens, d_inner)。

        子类实现应：
          1. 对每条序列从 cache 加载状态
          2. 调用 prefill / decode 路径
          3. 将更新后的状态写回 cache
        """
        pass

    def end_forward(self) -> None:
        """清理 forward 阶段的临时元数据。"""
        self.is_metadata_initialized  = False
        self.seq_is_prefill           = []
        self.seq_lens                 = []
        self.seq_state_block_ids      = []
        self.seq_token_offsets        = []

        