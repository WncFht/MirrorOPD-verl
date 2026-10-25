"""Keep verl's text-only tokenizer support while retaining vLLM's V2 runner."""

from types import MethodType

from transformers import AutoTokenizer
from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def mask_tokenizer_tail(model, width):
    original = model.compute_logits

    def compute_logits(self, *args, **kwargs):
        logits = original(*args, **kwargs)
        logits[..., width:] = -float("inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class ProtocolWorker(Worker):
    def __init__(self, vllm_config, *args, **kwargs):
        # verl assigns seed = rollout seed + replica rank. Dense vLLM DP resets
        # data_parallel_rank to zero but retains the actual data_parallel_index.
        replica = vllm_config.parallel_config.data_parallel_index
        vllm_config.model_config.seed += replica
        logger.info("MirrorOPD replica=%s sampling seed=%s", replica, vllm_config.model_config.seed)
        super().__init__(vllm_config, *args, **kwargs)

    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer, local_files_only=True)
        mask_tokenizer_tail(self.model_runner.get_model(), len(tokenizer))
