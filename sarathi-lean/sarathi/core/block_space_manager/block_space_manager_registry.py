from sarathi.config import SchedulerType
from sarathi.core.block_space_manager.vllm_block_space_manager import (
    VLLMBlockSpaceManager,
)
from sarathi.utils.base_registry import BaseRegistry


class BlockSpaceManagerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


BlockSpaceManagerRegistry.register(SchedulerType.VLLM, VLLMBlockSpaceManager)
