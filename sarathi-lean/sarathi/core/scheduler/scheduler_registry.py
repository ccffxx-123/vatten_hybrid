from sarathi.config import SchedulerType
from sarathi.core.scheduler.vllm_scheduler import VLLMScheduler
from sarathi.utils.base_registry import BaseRegistry


class SchedulerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


SchedulerRegistry.register(SchedulerType.VLLM, VLLMScheduler)
