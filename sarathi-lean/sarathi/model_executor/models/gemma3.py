# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py
# Copyright 2024 The Sarathi team.
# Copyright 2024 Google DeepMind and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""仅用于推理的Gemma3模型(文本部分),兼容HuggingFace权重。

模型的输入被展平为一维token张量。

Gemma3的核心特性(相比Gemma2):
1. 混合注意力模式: 每5层sliding window + 1层full attention (pattern=6)
2. 更小的滑动窗口: 1024 (Gemma2是4096)
3. 双RoPE频率: 全局注意力用rope_theta=1M, 本地注意力用rope_local_base_freq=10K
4. 128K上下文长度支持
5. 移除了attn_logit_softcapping (Gemma2有50.0)
6. 保留final_logit_softcapping (可选)
7. 继承Gemma2的Pre-norm + Post-norm混合归一化

配置类: Gemma3TextConfig
"""
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.model_executor.layers.activation import GeGLUActivation
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.layers.rotary_embedding import get_rope
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from sarathi.model_executor.parallel_utils.pipeline_parallel.mappings import recv, send
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from sarathi.model_executor.weight_utils import (
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)
from sarathi.worker.cache_engine import KVCache


class Gemma3MLP(nn.Module):
    """Gemma3的MLP模块
    
    使用GeGLU激活函数:
    - gate_proj和up_proj融合为gate_up_proj
    - 使用GELU激活,而非SiLU
    
    与Gemma2完全相同。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        # 门控上投影层(融合gate和up)
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        
        # 下投影层
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        
        # 验证激活函数
        if hidden_act not in ["gelu", "gelu_pytorch_tanh", "gelu_new"]:
            raise ValueError(
                f"不支持的激活函数: {hidden_act}. "
                "Gemma3应该使用gelu或gelu_pytorch_tanh。"
            )
        
        # GeGLU激活函数
        self.act_fn = GeGLUActivation(
            approximate="tanh" if "tanh" in hidden_act else "none"
        )

        # 性能计时器
        self._mlp_activation_timer = CudaTimer(
            OperationMetrics.MLP_ACTIVATION, layer_id=layer_id
        )

    def forward(self, x):
        """MLP前向传播: x -> gate_up -> GeGLU -> down -> output"""
        gate_up, _ = self.gate_up_proj(x)
        
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        
        x, _ = self.down_proj(x)
        return x


class Gemma3Attention(nn.Module):
    """Gemma3的注意力模块
    
    Gemma3的独特特性:
    1. 混合注意力: 每5层sliding window + 1层full attention
    2. 双RoPE频率: 全局用1M,本地用10K
    3. Query预缩放: 使用query_pre_attn_scalar
    4. 无attn_logit_softcapping (与Gemma2不同)
    
    注意力模式(sliding_window_pattern=6):
    - layer 0-4: sliding_attention (本地)
    - layer 5: full_attention (全局)
    - layer 6-10: sliding_attention
    - layer 11: full_attention
    - ...
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 131072,
        rope_theta: float = 1000000.0,  # 全局注意力的RoPE基频
        rope_local_base_freq: float = 10000.0,  # 本地注意力的RoPE基频
        rope_scaling: Optional[Dict[str, Any]] = None,
        attention_bias: bool = False,
        sliding_window: Optional[int] = 1024,  # Gemma3默认1024
        query_pre_attn_scalar: Optional[int] = 256,
        layer_type: str = "full_attention",
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.layer_type = layer_type
        self.sliding_window = sliding_window
        self.layer_id = layer_id
        
        # 张量并行配置
        tp_size = get_tensor_model_parallel_world_size()
        
        # 查询头分布
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        
        # KV头分布(GQA)
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            # KV头数少于TP数时,复制KV头
            assert tp_size % self.total_num_kv_heads == 0
            self.num_kv_heads = 1
        
        # Q、K、V的维度
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        # ========== Gemma3: Query预缩放 ==========
        # 标准缩放因子
        self.scaling = self.head_dim**-0.5
        
        # 使用query_pre_attn_scalar进行额外缩放
        if query_pre_attn_scalar is not None:
            self.query_pre_attn_scalar = query_pre_attn_scalar**-0.5
        else:
            self.query_pre_attn_scalar = self.head_dim**-0.5
        
        # ========== Gemma3: 双RoPE频率 ==========
        # 根据层类型选择不同的RoPE基频
        if layer_type == "full_attention":
            # 全局注意力使用高频率(1M)以支持长上下文
            effective_rope_theta = rope_theta
        else:
            # 本地/滑动窗口注意力使用低频率(10K)
            effective_rope_theta = rope_local_base_freq
        
        self.rope_theta = effective_rope_theta
        self.max_position_embeddings = max_position_embeddings

        # QKV投影层(融合)
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=attention_bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        
        # 输出投影层
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        
        # RoPE位置编码(使用层特定的基频)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        
        # 性能计时器
        self._attn_rope_timer = CudaTimer(
            OperationMetrics.ATTN_ROPE, layer_id=layer_id
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """Gemma3注意力前向传播
        
        特殊处理:
        1. Query预缩放
        2. 滑动窗口注意力 (根据layer_type)
        3. 无soft-capping (与Gemma2不同)
        
        Args:
            positions: 位置索引
            hidden_states: 输入隐藏状态
            kv_cache: KV缓存
            
        Returns:
            注意力输出
        """
        # 1. QKV投影
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 2. 应用RoPE位置编码
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)
        
        # 3. Query预缩放
        q = q * self.query_pre_attn_scalar
        
        # 4. 执行注意力计算
        # Gemma3没有attn_logit_softcapping
        attn_output = get_attention_wrapper().forward(
            q,
            k,
            v,
            kv_cache,
            self.scaling,
            self.layer_id,
            attention_type=self.layer_type,
            sliding_window=self.sliding_window if self.layer_type == "sliding_attention" else None,
        )
        
        # 5. 输出投影
        output, _ = self.o_proj(attn_output)
        return output


class Gemma3DecoderLayer(nn.Module):
    """Gemma3解码器层
    
    与Gemma2相同,使用混合归一化策略:
    1. Pre-norm: attention和MLP前的LayerNorm
    2. Post-norm: attention和MLP后的额外LayerNorm
    
    结构:
    x = x + Attention(Pre-LN(x))
    x = Post-LN(x)
    x = x + MLP(Pre-LN(x))
    x = Post-LN(x)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_type: str = "full_attention",
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = layer_type
        
        # 获取配置参数
        head_dim = getattr(config, "head_dim", 256)
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        rope_local_base_freq = getattr(config, "rope_local_base_freq", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        attention_bias = getattr(config, "attention_bias", False)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        sliding_window = getattr(config, "sliding_window", 1024)
        query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", 256)
        
        # 获取隐藏层激活函数名称
        hidden_act = getattr(config, "hidden_act", None)
        if hidden_act is None:
            hidden_act = getattr(config, "hidden_activation", "gelu_pytorch_tanh")
        
        # 自注意力层
        self.self_attn = Gemma3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_local_base_freq=rope_local_base_freq,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            sliding_window=sliding_window,
            query_pre_attn_scalar=query_pre_attn_scalar,
            layer_type=layer_type,
            layer_id=layer_id,
        )
        
        # MLP层
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=hidden_act,
            layer_id=layer_id,
        )
        
        # 获取RMS norm epsilon
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        
        # ========== 4个LayerNorm (与Gemma2相同) ==========
        # Pre-norm for attention
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=rms_norm_eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )
        
        # Post-norm for attention
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=rms_norm_eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
        )
        
        # Pre-norm for MLP
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=rms_norm_eps,
            layer_id=layer_id,
        )
        
        # Post-norm for MLP
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size,
            eps=rms_norm_eps,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """Gemma3解码器层前向传播
        
        混合归一化:
        1. Pre-norm + Attention + 残差
        2. Post-norm
        3. Pre-norm + MLP + 残差
        4. Post-norm
        """
        # ========== 自注意力块 ==========
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # ========== MLP块 ==========
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        
        return hidden_states


def _get_gemma3_layer_types(
    num_hidden_layers: int,
    sliding_window_pattern: int = 6,
) -> List[str]:
    """生成Gemma3的layer_types列表
    
    Gemma3的注意力模式:
    - sliding_window_pattern=6 表示每6层中有5层sliding + 1层full
    - 具体: (i+1) % pattern != 0 的层使用sliding_attention
    - 例如pattern=6: layers 0-4用sliding, layer 5用full, 以此类推
    
    Args:
        num_hidden_layers: 总层数
        sliding_window_pattern: 注意力模式周期
        
    Returns:
        每层的注意力类型列表
    """
    layer_types = []
    for i in range(num_hidden_layers):
        # (i+1) % pattern == 0 的层使用full_attention
        if (i + 1) % sliding_window_pattern == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("sliding_attention")
    return layer_types


class Gemma3TextModel(nn.Module):
    """Gemma3文本模型主体
    
    完整的Gemma3 Transformer模型(仅文本部分)。
    
    特性:
    - 混合注意力: 每5层sliding window + 1层full attention
    - 128K上下文长度
    - 双RoPE频率
    - 嵌入缩放因子
    """

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Gemma嵌入缩放因子
        self.embed_scale = config.hidden_size ** 0.5

        # Token嵌入层(仅第一阶段)
        self.embed_tokens = None
        if is_pipeline_first_stage():
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size,
                config.hidden_size,
                perform_initialization=False,
                linear_metric_name=OperationMetrics.EMBED_LINEAR,
                communication_metric_name=OperationMetrics.EMBED_ALL_REDUCE,
            )

        # 解码器层
        num_layers = (
            config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        )
        layer_offset = get_pipeline_model_parallel_rank() * num_layers
        
        # 获取layer_types配置
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            # 如果没有指定,根据sliding_window_pattern生成
            sliding_window_pattern = getattr(config, "sliding_window_pattern", 6)
            # 兼容私有属性
            if sliding_window_pattern is None:
                sliding_window_pattern = getattr(config, "_sliding_window_pattern", 6)
            layer_types = _get_gemma3_layer_types(
                config.num_hidden_layers,
                sliding_window_pattern,
            )
        
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(
                    config,
                    layer_type=layer_types[layer_id + layer_offset],
                    layer_id=layer_id + layer_offset,
                )
                for layer_id in range(num_layers)
            ]
        )

        # 最终层归一化(仅最后阶段)
        self.norm = None
        if is_pipeline_last_stage():
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
            self.norm = RMSNorm(
                config.hidden_size,
                eps=rms_norm_eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> torch.Tensor:
        """Gemma3模型前向传播"""
        # Token嵌入(仅第一阶段)
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)
            # Gemma嵌入缩放
            hidden_states = hidden_states * self.embed_scale

        # 通过所有解码器层
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                positions,
                hidden_states,
                kv_caches[i],
            )

        # 最终层归一化(仅最后阶段)
        if self.norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    """用于因果语言建模的Gemma3模型
    
    完整的语言模型,包含:
    1. Gemma3文本模型主体
    2. 语言模型头(LM head)
    3. Final logit soft-capping (可选)
    4. 流水线并行通信
    """

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        
        # Gemma3文本模型主体
        self.model = Gemma3TextModel(config)
        
        vocab_size = ((config.vocab_size + 63) // 64) * 64

        # 流水线阶段标记
        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        # ========== Final logit soft-capping (可选) ==========
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)

        # 语言模型头(仅最后阶段)
        self.lm_head = None
        if self.is_pipeline_last_stage:
            lm_head_bias = getattr(config, "lm_head_bias", False)
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                bias=lm_head_bias,
                gather_output=False,
                perform_initialization=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
    ) -> torch.Tensor:
        """Gemma3语言模型前向传播"""
        # 流水线并行: 接收阶段
        if not self.is_pipeline_first_stage:
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        # 通过Gemma3模型
        hidden_states = self.model(hidden_states, positions, kv_caches)

        # 流水线并行: 发送阶段
        if not self.is_pipeline_last_stage:
            send(hidden_states)
            return hidden_states

        # ========== LM head和Final soft-capping ==========
        if self.lm_head is not None:
            logits, _ = self.lm_head(hidden_states)
            
            # Final logit soft-capping (如果启用)
            if self.final_logit_softcapping is not None:
                logits = logits / self.final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * self.final_logit_softcapping
            
            return logits

        return hidden_states

    # 权重加载配置
    _column_parallel_layers = []
    _row_parallel_layers = ["o_proj", "down_proj"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        """从HuggingFace格式加载Gemma3权重
        
        权重名称格式:
        - model.embed_tokens.weight
        - model.layers.{i}.self_attn.q_proj.weight
        - model.layers.{i}.self_attn.k_proj.weight
        - model.layers.{i}.self_attn.v_proj.weight
        - model.layers.{i}.self_attn.o_proj.weight
        - model.layers.{i}.mlp.gate_proj.weight
        - model.layers.{i}.mlp.up_proj.weight
        - model.layers.{i}.mlp.down_proj.weight
        - model.layers.{i}.input_layernorm.weight
        - model.layers.{i}.post_attention_layernorm.weight
        - model.layers.{i}.pre_feedforward_layernorm.weight
        - model.layers.{i}.post_feedforward_layernorm.weight
        - model.norm.weight
        - lm_head.weight (如果不tie_word_embeddings)
        """
        weight_suffixes = ["weight"]

        # 构建并行层名称列表
        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        # 获取并行配置
        tp_size = get_tensor_model_parallel_world_size()
        pp_size = get_pipeline_model_parallel_world_size()
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        pp_model_parallel_rank = get_pipeline_model_parallel_rank()

        # 计算层范围
        assert self.config.num_hidden_layers % pp_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_size

        first_layer_id = layers_per_stage * pp_model_parallel_rank
        last_layer_id = layers_per_stage * (pp_model_parallel_rank + 1) - 1

        # 计算QKV分片大小
        head_dim = getattr(self.config, "head_dim", 256)
        q_proj_shard_size = self.config.num_attention_heads * head_dim // tp_size
        
        # 处理KV头数少于TP数的情况
        num_kv_heads = self.config.num_key_value_heads
        if num_kv_heads >= tp_size:
            kv_proj_shard_size = num_kv_heads * head_dim // tp_size
        else:
            # KV头数少于TP数时,每个rank复制全部KV头
            kv_proj_shard_size = num_kv_heads * head_dim
        
        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        
        state_dict = self.state_dict()

        # 遍历并加载权重
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            # 跳过RoPE逆频率
            if "rotary_emb.inv_freq" in name:
                continue

            # 跳过非当前阶段的嵌入层
            if pp_model_parallel_rank != 0 and "embed_tokens" in name:
                continue

            # 跳过非当前阶段的LM头和归一化
            if pp_model_parallel_rank != pp_size - 1 and (
                "lm_head" in name or name == "model.norm.weight"
            ):
                continue

            # 处理解码器层权重
            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue

                new_layer_id = layer_id - first_layer_id
                name = name.replace(f"layers.{layer_id}.", f"layers.{new_layer_id}.")

            # QKV权重融合
            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name or "self_attn" not in name:
                    continue
                
                param = state_dict[name.replace(weight_name, "qkv_proj")]

                # 处理KV头复制
                if weight_name in ["k_proj", "v_proj"] and num_kv_heads < tp_size:
                    # 复制KV头到所有rank
                    loaded_weight_shard = loaded_weight
                else:
                    loaded_weight_shard = loaded_weight[
                        shard_size * tensor_model_parallel_rank : 
                        shard_size * (tensor_model_parallel_rank + 1)
                    ]
                
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight_shard.shape, \
                    f"Shape mismatch: {param_slice.shape} vs {loaded_weight_shard.shape}"

                param_slice.copy_(loaded_weight_shard)
                is_attention_weight = True
                break
            
            if is_attention_weight:
                continue

            # Gate-Up权重融合
            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name or "mlp" not in name:
                    continue
                
                param = state_dict[name.replace(weight_name, "gate_up_proj")]

                shard_size = param.shape[0] // 2
                loaded_weight_shard = loaded_weight[
                    shard_size * tensor_model_parallel_rank : 
                    shard_size * (tensor_model_parallel_rank + 1)
                ]
                param_slice = param.data[
                    shard_size * stride_id : shard_size * (stride_id + 1)
                ]
                assert param_slice.shape == loaded_weight_shard.shape
                param_slice.copy_(loaded_weight_shard)
                is_gate_up_weight = True
                break
            
            if is_gate_up_weight:
                continue

            # 检查权重是否存在于state_dict中
            if name not in state_dict:
                print(f"Warning: weight {name} not found in state_dict, skipping...")
                continue

            # 标准权重加载
            param = state_dict[name]

            # 嵌入层和LM头
            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(
                    param, loaded_weight, tensor_model_parallel_rank
                )
                continue

            # 通用张量并行权重
            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tensor_model_parallel_rank,
            )


# 为了兼容性,提供别名
Gemma3TextForCausalLM = Gemma3ForCausalLM