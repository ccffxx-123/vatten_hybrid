from sarathi.core.datatypes.sequence import Sequence
import torch
from typing import Dict, List
import vattention
from sarathi.worker.cache_engine import get_cache_engine
from sarathi.model_executor.attention import get_attn_type
import math
from sarathi.config import ModelConfig, ParallelConfig

# 按照 block 管理
class vAttentionBlockSpaceManager():

    def __init__(self, 
        block_size: int,
        num_gpu_blocks: int,
        max_model_len: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size 
        self.num_total_gpu_blocks = num_gpu_blocks
        self.max_model_len = max_model_len
        self.promised_blocks = 0
        self.watermark = watermark
        assert watermark >= 0.0
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.active_requests: Dict[int, Sequence] = {}
        self.preemption_queue = []

    # def reset_free_blocks():
    #     self.free_blocks = 0

    def get_num_blocks(self, seq: Sequence) -> int:
        # print("seq.get_len(): ", seq.get_len(), " self.block_size: ", self.block_size)
        len_seq = seq.get_len()
        num_blocks = math.ceil(len_seq / self.block_size)
        return num_blocks

    def can_allocate(self, seq: Sequence) -> bool:
        # return True
        # if self.__getattribute__('free_blocks') is None:
        #     return True
        num_required_blocks = self.get_num_blocks(seq)
        num_free_gpu_blocks = self.free_blocks
        # print("num_free_gpu_blocks: ", num_free_gpu_blocks, " num_required_blocks: ", num_required_blocks, " self.promised_blocks: ", self.promised_blocks, " self.watermark_blocks: ", self.watermark_blocks)
        return num_free_gpu_blocks - self.promised_blocks - num_required_blocks >= self.watermark_blocks

    def set_free_blocks(self, free_blocks: int) -> None:
        self.free_blocks = free_blocks

    def allocate(self, seq: Sequence) -> None:
        self.active_requests[seq.seq_id] = seq
        self.promised_blocks += self.get_num_blocks(seq)
    
    def can_append_slot(self) -> bool:
        # num_free_gpu_blocks = self.free_blocks
        # return (num_free_gpu_blocks - self.promised_blocks) > 0
        # return True
        # return self.free_blocks > self.promised_blocks *1.1
        return self.free_blocks - self.promised_blocks > 0
        

    def append_slot(self, seq: Sequence) -> None:
        """Allocate a physical slot for a new token."""
        len_seq = seq.get_len()
        num_blocks_current = math.ceil(len_seq / self.block_size)
        num_blocks_new = math.ceil((len_seq + 1) / self.block_size)
        if num_blocks_new > num_blocks_current:
            self.promised_blocks += 1
        # pass

    def _get_physical_blocks(self, seq: Sequence):
        pass

    def _free_block_table(self, block_table) -> None:
        pass

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.active_requests:
            # Already freed or haven't been scheduled yet.
            return
        else:
            del self.active_requests[seq.seq_id]
            self.free_blocks += self.get_num_blocks(seq)

    def reset(self) -> None:
        self.active_requests = {}
        pass

    def clear_promised_blocks(self) -> None:
        self.promised_blocks = 0

    def get_block_table(self, seq: Sequence) -> List[int]:
        pass

    def is_allocated(self, seq: Sequence) -> bool:
        return seq.seq_id in self.active_requests
    
    def get_num_free_gpu_blocks(self, seq: Sequence) -> int:
        return self.free_blocks