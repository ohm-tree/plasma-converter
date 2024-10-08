# Set up vllm stuff.
import gc

from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from src.workers.worker import *

# detect the type of gpus available.
# If there is at least one A100 80GB or H100,
# tensor parallelization is not needed.
# Else, check that there are at least 4 V100s.
# and set tensor_parallel_size=4

# num_devices = torch.cuda.device_count()
# device_counts = {}
# for i in range(num_devices):
#     device_name = torch.cuda.get_device_name(i)
#     if device_name in device_counts:
#         device_counts[device_name] += 1
#     else:
#         device_counts[device_name] = 1

# print(f"Worker {completion_worker_id} detected the following devices: {device_counts}")

# if "A100-SXM4-80GB" in device_counts or "H100-SXM4-80GB" in device_counts:
#     tensor_parallel_size = 1
#     llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
#               max_num_batched_tokens=8192,
#               trust_remote_code=True
#               )
# elif "Tesla V100-SXM2-16GB" in device_counts:
#     if device_counts["Tesla V100-SXM2-16GB"] >= 4:
#         tensor_parallel_size = 4
#     else:
#         raise ValueError("Not enough Tesla V100-SXM2-16GB GPUs available.")


class LLMWorker(Worker):
    def __init__(self,
                 worker_id: WorkerIdentifer,
                 queues: Dict[Union[WorkerType, WorkerIdentifer], multiprocessing.Queue],
                 run_name: str,
                 gpu_set: List[int],
                 model_name: Optional[str] = "deepseek-ai/DeepSeek-Prover-V1.5-RL",
                 LLM_kwargs: Optional[dict] = None,
                 sampling_kwargs: Optional[dict] = None,
                 use_tqdm: bool = True  # Dopamine
                 ):
        super().__init__(worker_id, queues, run_name)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_set))

        self.llm: LLM
        if LLM_kwargs is None:
            self.llm = LLM(
                model=model_name,
                max_num_batched_tokens=8192,
                trust_remote_code=True,
                tensor_parallel_size=len(gpu_set)
            )
        else:
            LLM_kwargs['tensor_parallel_size'] = len(gpu_set)
            self.llm = LLM(
                **LLM_kwargs
            )

        if sampling_kwargs is None:
            self.sampling_params = SamplingParams(
                max_tokens=1024,
                temperature=0.0,
                top_k=1,
                top_p=1.0
            )
        else:
            self.sampling_params = SamplingParams(
                **sampling_kwargs
            )

        self.gpu_set = gpu_set

        self.use_tqdm = use_tqdm

    def generate(self, input_data: List[str]) -> List[RequestOutput]:
        return self.llm.generate(
            input_data,
            sampling_params=self.sampling_params,
            use_tqdm=self.use_tqdm
        )

    def shutdown(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm
        gc.collect()
