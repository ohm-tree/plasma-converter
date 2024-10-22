# Set up vllm stuff.
import gc
from secrets import token_bytes

# remote queries to openai
from openai import OpenAI
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
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: dict[str, multiprocessing.Queue],
                 gpu_set: list[int],
                 use_tqdm: bool = True,  # Dopamine
                 **kwargs  # Unused
                 ):
        super().__init__(
            name="LLM" + "_" + str(task_id),
            worker_type="LLM",
            worker_idx=task_id,
            queues=queues,
            run_name=run_name,
        )
        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )

        LLM_kwargs = config['model'] if 'model' in config else None
        sampling_kwargs = config['sampling'] if 'sampling' in config else None

        self.channel = config['channel']  # Required.

        if LLM_kwargs is None:
            LLM_kwargs = {
                "model": "deepseek-ai/DeepSeek-Prover-V1.5-RL",
                "max_num_batched_tokens": 8192,
                "trust_remote_code": True,
                "tensor_parallel_size": len(gpu_set)
            }
        else:
            LLM_kwargs['tensor_parallel_size'] = len(gpu_set)
        self.LLM_kwargs = LLM_kwargs
        if sampling_kwargs is None:
            sampling_kwargs = {
                "max_tokens": 1024,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0
            }

        self.sampling_params = SamplingParams(
            **sampling_kwargs
        )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_set))

        self.llm = LLM(
            **LLM_kwargs
        )
        self.gpu_set = gpu_set
        self.use_tqdm = use_tqdm

    def generate(self, input_data: Union[list[str], list[dict]]) -> list[RequestOutput]:
        return self.llm.generate(
            input_data,
            sampling_params=self.sampling_params,
            use_tqdm=self.use_tqdm
        )

    def loop(self):
        my_tasks = self.spin_deque_tasks(
            channel=self.channel,
            timeout=self.config['timeout'],
            batch_size=self.config['batch_size'],
        )
        self.logger.info(
            f"Received {len(my_tasks)} tasks.")

        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            return
        # We have tasks to complete.
        input_data = [i['prompt'] for i in my_tasks]
        outputs: list[RequestOutput] = self.generate(input_data)
        for i in range(len(outputs)):
            text = outputs[i].outputs[0].text
            token_ids = outputs[i].outputs[0].token_ids
            cumulative_logprob = outputs[i].outputs[0].cumulative_logprob
            self.logger.info(text)
            self.logger.info(token_ids)
            self.logger.info(cumulative_logprob)

            response = {
                'text': text,
                'token_ids': token_ids,
                'cumulative_logprob': cumulative_logprob
            }

            self.enqueue(
                response=response,
                channel=my_tasks[i]['channel']  # The response channel.
            )

    def shutdown(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm
        gc.collect()
