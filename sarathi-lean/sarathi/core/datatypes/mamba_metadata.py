from typing import List

# 在 model_runner.py 顶部定义
class MambaInputMetadata:
    def __init__(
        self,
        seq_is_prefill: List[bool],
        seq_lens: List[int],
        seq_state_block_ids: List[int],
        seq_token_offsets: List[int],
    ):
        self.seq_is_prefill = seq_is_prefill
        self.seq_lens = seq_lens
        self.seq_state_block_ids = seq_state_block_ids
        self.seq_token_offsets = seq_token_offsets
