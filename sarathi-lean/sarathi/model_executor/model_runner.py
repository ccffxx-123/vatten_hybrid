from typing import List, Optional, Tuple, Dict

import torch
import torch.distributed

from sarathi.config import (
    BaseSchedulerConfig,
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerType,
)
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence import Sequence, SequenceMetadata
from sarathi.logger import init_logger
from sarathi.metrics.constants import CpuOperationMetrics, OperationMetrics
from sarathi.metrics.cpu_timer import CpuTimer
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor import get_model, set_random_seed
from sarathi.model_executor.attention import get_attention_wrapper
from sarathi.model_executor.layers.sampler import Sampler
from sarathi.model_executor.utils import pad_to_alignment
from sarathi.utils import get_gpu_memory
from sarathi.worker.cache_engine import get_cache_engine
from sarathi.model_executor.attention import AttentionBackend
from sarathi.model_executor.attention import get_mamba_wrapper
from sarathi.model_executor.attention import AttentionBackend
from sarathi.core.datatypes.mamba_metadata import MambaInputMetadata
logger = init_logger(__name__)


def _build_layer_to_group_idx(model_config: ModelConfig) -> Dict[int, int]:
    """
    从 ModelConfig 推导每个 Mamba 层（全局真实 layer_id）对应的 group_idx。
 
    关键点
    ------
    model_config.get_layer_type_list() 会过滤掉 "mlp" 类型层并重新编号，
    返回的索引与模型中真实的 global_layer_id（含 mlp 层）不一致。
 
    例：layers_block_type = ["mamba","mlp","mamba","mlp","mamba",...]
      get_layer_type_list() 返回 ["state","state","state",...]，索引 0,1,2,...
      但真实 global_layer_id 是 0,2,4,...
 
    因此本函数直接读取 hf_config.layers_block_type（原始完整列表），
    用真实全局 layer_id 构建映射，与 NemotronHBlock.layer_id 保持一致。
 
    分组算法复现 build_kv_cache_config() 的逻辑：
      _TYPE_ORDER = ("trans", "swa", "state")
      group_size  = min(各非 mlp 类型的层数)
      按 _TYPE_ORDER 顺序切块，每块对应一个 group，group 编号全局累加
 
    返回
    ----
    Dict[real_global_layer_id -> group_idx]
        只包含 "state"（mamba）类型层的条目。
    """
    _TYPE_ORDER = ("trans", "swa", "state")
 
    # hf_config.layers_block_type 是含 mlp 的原始完整列表
    # 例：["mamba","mlp","mamba","mlp","mamba","attention","mlp",...]
    hf_config = model_config.hf_config
    raw_block_types = getattr(hf_config, "layers_block_type", None)
 
    if raw_block_types is not None:
        real_ids_with_type = list(enumerate(raw_block_types))
    else:
        # fallback：模型不含 mlp 层，get_layer_type_list() 的索引即真实 global_id
        real_ids_with_type = list(enumerate(model_config.get_layer_type_list()))
 
    # 将 hf_config 的类型名映射到 sarathi 内部类型名
    _HF_TO_SARATHI = {
        "mamba":          "state",
        "ssm":            "state",
        "recurrent":      "state",
        "attention":      "trans",
        "full_attention": "trans",
        "transformer":    "trans",
        "sliding_window": "swa",
        "mlp":            "mlp",   # 不参与分组，跳过
    }
 
    # 按 sarathi 类型收集真实 global_layer_id（跳过 mlp）
    layer_ids_by_type: Dict[str, List[int]] = {"trans": [], "swa": [], "state": []}
    for real_id, raw_type in real_ids_with_type:
        sarathi_type = _HF_TO_SARATHI.get(raw_type.lower())
        if sarathi_type is None or sarathi_type == "mlp":
            continue
        layer_ids_by_type[sarathi_type].append(real_id)
 
    # 只保留实际存在的类型
    present_types = {t for t, ids in layer_ids_by_type.items() if ids}
 
    # group_size = min(各存在类型的层数)
    layer_counts = {t: len(layer_ids_by_type[t]) for t in present_types}
    group_size = min(layer_counts.values())
 
    # 按 _TYPE_ORDER 切块，累计 group 编号，只记录 "state" 层的映射
    layer_to_group_idx: Dict[int, int] = {}
    group_counter = 0
 
    for sarathi_type in _TYPE_ORDER:
        if sarathi_type not in present_types:
            continue
        ids = layer_ids_by_type[sarathi_type]
        for chunk_start in range(0, len(ids), group_size):
            chunk = ids[chunk_start: chunk_start + group_size]
            if sarathi_type == "state":
                for real_id in chunk:
                    layer_to_group_idx[real_id] = group_counter
            group_counter += 1
 
    return layer_to_group_idx


USE_UVM = False
class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        device: torch.device,
        rank: int,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device = device
        self.rank = rank
        self.cache_config = cache_config

        self.model = get_model(self.model_config)
        get_attention_wrapper().init(
            self.model_config,
            self.parallel_config,
            cache_config.block_size,
            self.device,
        )
        self.has_mamba_layers = (
            any(t == "state" for t in model_config.get_layer_type_list())
        )

        if self.has_mamba_layers:
            # 构建 global_layer_idx → group_idx 映射
            layer_to_group_idx = _build_layer_to_group_idx(model_config)
            logger.info(
                f"[ModelRunner] Mamba layer → group_idx mapping: {layer_to_group_idx}"
            )
            # 将映射注入模型中每个 Mamba 层
            # 模型负责识别自己的层并设置 group_idx（见 NemotronHModel.set_mamba_group_indices）
            if hasattr(self.model, "set_mamba_group_indices"):
                self.model.set_mamba_group_indices(layer_to_group_idx)
            else:
                logger.warning(
                    "[ModelRunner] model 没有 set_mamba_group_indices 方法，"
                    "Mamba 层的 group_idx 未设置，block_table 路由可能错误。"
                )


        self.sampler: Optional[Sampler] = None
        if self.model.lm_head:
            self.sampler = Sampler(
                self.model.lm_head.weight, self.model.config.vocab_size
            )

        self._prepare_inputs_e2e_timer = CpuTimer(
            CpuOperationMetrics.PREPARE_INPUTS_E2E, rank=self.rank
        )
        self._sampler_e2e_timer = CpuTimer(
            CpuOperationMetrics.SAMPLER_E2E, rank=self.rank
        )
        self._model_execution_e2e_timer = CpuTimer(
            CpuOperationMetrics.MODEL_EXECUTION_E2E, rank=self.rank
        )

        

    def _prepare_inputs(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        # need to know prompt chunk sizes for each prompt sequence for sampler
        current_prompt_chunk_lens: List[int] = []

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            current_prompt_chunk_tokens = (
                seq_metadata.seq.get_next_prompt_chunk_token_ids(prompt_chunk_len)
            )
            current_prompt_chunk_len = len(current_prompt_chunk_tokens)
            current_prompt_chunk_lens.append(current_prompt_chunk_len)
            processed_prompt_len = seq_metadata.seq.get_num_prompt_tokens_processed()

            current_total_len = processed_prompt_len + current_prompt_chunk_len

            input_tokens.extend(current_prompt_chunk_tokens)
            input_positions.extend(range(processed_prompt_len, current_total_len))

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            generation_token = seq_metadata.seq.get_last_token_id()
            input_tokens.append(generation_token)

            context_len = seq_metadata.seq.get_len()
            position = context_len - 1
            input_positions.append(position)
        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
        positions_tensor = torch.tensor(
            input_positions, dtype=torch.long, device=self.device
        )

        return tokens_tensor, positions_tensor


    def _build_mamba_metadata(
        self,
        seq_metadata_list: List[SequenceMetadata],
        is_profiling: bool = False,
    ) -> MambaInputMetadata:
        """
        构建 MambaInputMetadata。
 
        is_profiling=True 时：
          - seq_metadata_list 中的 block_tables 为 None
          - seq_metadata_list 传 None，层内按 block_tables is None 跳过 cache 读写
 
        is_profiling=False 时：
          - 传完整 seq_metadata_list，每个 Mamba 层按自己的 group_idx 查 block_tables
        """
        seq_is_prefill: List[bool] = []
        seq_lens: List[int] = []
        seq_token_offsets: List[int] = []
        token_offset = 0
 
        for seq_meta in seq_metadata_list:
            is_prefill = seq_meta.is_prompt
            if is_prefill:
                seq_len = seq_meta.seq.get_next_prompt_chunk_len(
                    seq_meta.prompt_chunk_len
                )
            else:
                seq_len = 1
 
            seq_is_prefill.append(is_prefill)
            seq_lens.append(seq_len)
            seq_token_offsets.append(token_offset)
            token_offset += seq_len
 
        return MambaInputMetadata(
            seq_is_prefill=seq_is_prefill,
            seq_lens=seq_lens,
            seq_token_offsets=seq_token_offsets,
            seq_metadata_list=None if is_profiling else seq_metadata_list,
        )


    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int]:
        torch.cuda.set_device(self.device)

        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        seq_metadata_list: List[SequenceMetadata] = []
   
        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        for seq_id in range(max_num_seqs):
            seq_len = max_num_batched_tokens // max_num_seqs + (
                seq_id < max_num_batched_tokens % max_num_seqs
            )

            seq = Sequence(
                seq_id=seq_id,
                prompt=None,
                prompt_token_ids=[0] * seq_len,
                block_size=block_size,
                eos_token_id=1,
                arrival_time=None,
                sampling_params=sampling_params,
            )
            seq_metadata = SequenceMetadata(
                seq=seq,
                block_table=None,
                prompt_chunk_len=seq_len,
            )
            seq_metadata_list.append(seq_metadata)

        input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)

        get_attention_wrapper().begin_forward(seq_metadata_list)

        # -------------------- 新增：Mamba Profiling 元数据构造 --------------------
        input_metadata = None
        if self.has_mamba_layers:
            # profiling 阶段：seq_metadata_list=None，层内跳过 cache 读写
            input_metadata = self._build_mamba_metadata(
                seq_metadata_list, is_profiling=True
            )
        # -----------------------------------------------------------------------

        if AttentionBackend.is_vATTN(self.model_config.attention_backend):
            get_attention_wrapper().is_profiling_iteration = True
        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        if self.has_mamba_layers:
            self.model(
                hidden_states=input_tokens,
                positions=input_positions,
                kv_caches=[None] * num_layers,
                input_metadata=input_metadata, # <--- 补上组装好的虚拟参数
            )
        else:
            self.model(
                hidden_states=input_tokens,
                positions=input_positions,
                kv_caches=[None] * num_layers,
            )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        # print("11111111111111111111111111111111111111111111111111111111111111111111111")
        # print(f"peak_memory: {peak_memory}, total_gpu_memory: {total_gpu_memory}")
        physical_memory = int(total_gpu_memory * gpu_memory_utilization - peak_memory)


        if self.model_config.is_hybrid_model() and AttentionBackend.is_vLLM_hybird(self.model_config.attention_backend):
            from sarathi.core.kv_cache_config_builder import build_kv_cache_config
            from sarathi.worker.cache_engine.hybrid_cache_engine import HybridCacheEngine
            # num_blocks=1 仅用于计算单 block 字节数
            template_config = build_kv_cache_config(
                self.model_config, self.cache_config,
                self.parallel_config, num_blocks=1,
            )
            cache_block_size = HybridCacheEngine.get_cache_block_size(template_config)
        else:
            cache_block_size = get_cache_engine(self.model_config.attention_backend).get_cache_block_size(
                block_size, self.model_config, self.parallel_config
            )

        logger.info(f"# cache_block_size: {cache_block_size}, physical_memory: {physical_memory}")
        num_gpu_blocks = int(physical_memory // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        torch.cuda.empty_cache()
        get_attention_wrapper().end_forward()
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, physical_memory
        

    def run(
        self,
        seq_metadata_list: List[SequenceMetadata],
        gpu_cache: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Prepare input tensors.
        with self._prepare_inputs_e2e_timer:
            input_tokens, input_positions = self._prepare_inputs(seq_metadata_list)

        get_attention_wrapper().begin_forward(seq_metadata_list)

        input_metadata = None
        if self.has_mamba_layers:
            input_metadata = self._build_mamba_metadata(
                seq_metadata_list, is_profiling=False
            )

        with self._model_execution_e2e_timer:
            # Execute the model.
            try:
                if self.has_mamba_layers:
                    output = self.model(
                        hidden_states=input_tokens,
                        positions=input_positions,
                        kv_caches=gpu_cache,
                        input_metadata=input_metadata,  # 3. 将元数据透传给模型底层
                    )
                else:
                    output = self.model(
                        hidden_states=input_tokens,
                        positions=input_positions,
                        kv_caches=gpu_cache,
                    )
            except RuntimeError as e:
                logger.error(
                    f"RuntimeError: {e} for seq_metadata_list: {seq_metadata_list}"
                )
                raise e

        with self._sampler_e2e_timer:
            if self.sampler is not None:
                output = self.sampler(output, seq_metadata_list)

        get_attention_wrapper().end_forward()

        return output
