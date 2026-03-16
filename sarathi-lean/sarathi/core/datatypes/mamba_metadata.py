from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sarathi.core.datatypes.sequence import SequenceMetadata


class MambaInputMetadata:
    """
    Mamba 层在 forward 时需要的批次元数据。

    字段说明
    --------
    seq_is_prefill    : 每条序列是否处于 prefill 阶段
    seq_lens          : 每条序列本次处理的 token 数
    seq_token_offsets : 每条序列在 hidden_states 中的起始位置
    seq_metadata_list : 原始 SequenceMetadata 列表，供每个 Mamba 层
                        按自己的 group_idx 查询正确的 block_table。
                        profiling 阶段传 None，层内按 block_tables is None 跳过 cache 读写。
    """

    def __init__(
        self,
        seq_is_prefill: List[bool],
        seq_lens: List[int],
        seq_token_offsets: List[int],
        seq_metadata_list: Optional[List["SequenceMetadata"]] = None,
    ):
        self.seq_is_prefill = seq_is_prefill
        self.seq_lens = seq_lens
        self.seq_token_offsets = seq_token_offsets
        self.seq_metadata_list = seq_metadata_list