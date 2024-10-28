# Set up vllm stuff.
import gc
from secrets import token_bytes

# remote queries to openai
from openai import OpenAI
from ray import worker
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
                 worker_type: str,
                 task_id: int,
                 queues: dict[str, multiprocessing.Queue],
                 gpu_set: list[int],
                 use_tqdm: bool = False,  # Dopamine
                 **kwargs  # Unused
                 ):
        super().__init__(
            # name="LLM" + "_" + str(task_id),
            # worker_type="LLM",
            name=worker_type + "_" + str(task_id),
            worker_type=worker_type,
            worker_idx=task_id,
            queues=queues,
            run_name=run_name,
        )
        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )
        self.config = config

        LLM_kwargs = config['model'] if 'model' in config else None
        sampling_kwargs = config['sampling'] if 'sampling' in config else None

        self.channel = self.worker_type

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
                "top_p": 1.0,
                "logprobs": 0,  # Return the logprobs
            }

        self.sampling_kwargs = sampling_kwargs

        # self.sampling_params = SamplingParams(
        #     **sampling_kwargs
        # )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_set))

        print("gpu_set: ", len(gpu_set), gpu_set)
        self.llm = LLM(
            **LLM_kwargs
        )
        self.gpu_set = gpu_set
        self.use_tqdm = use_tqdm

    def generate(self, input_data: list[str],
                 sampling_params: list[SamplingParams],
                 ) -> list[RequestOutput]:
        # self.logger.info(
        #     "input_data: " + str(input_data) + "\n" +
        #     "sampling_params: " + str(sampling_params)
        # )
        return self.llm.generate(
            input_data,
            sampling_params=sampling_params,
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

        sampling_param_keys = [
            'n',
            'temperature',
        ]
        sampling_kwarg_inputs = [
            {k: i[k] for k in i.keys() if k in sampling_param_keys}
            for i in my_tasks
        ]

        # take the union of all keys
        sampling_kwarg_inputs = [
            dict(
                self.sampling_kwargs,
                **i  # resolves in favor of i.
            )
            for i in sampling_kwarg_inputs
        ]
        # self.logger.info("Sampling kwargs:")
        # self.logger.info(sampling_kwarg_inputs)
        sampling_param_inputs = [
            SamplingParams(
                **i
            )
            for i in sampling_kwarg_inputs
        ]

        def truncate(snippet: str) -> str:
                new_code = snippet
                if new_code.endswith('```'):
                    new_code = new_code[:-3]

                lines = new_code.split('\n')
                def start_of_comment(line: str) -> bool:
                    # get the first character that's not a space
                    strip_str = line.lstrip()
                    first_non_space = strip_str[0] if strip_str else ''
                    return first_non_space in ['/', '-']
                for i in range(1, len(lines)):
                    if start_of_comment(lines[i]) and not start_of_comment(lines[i-1]):
                        new_code = '\n'.join(lines[:i]) + '\n'
                        break
                return new_code

        # sampling_param_inputs = [i['sampling_params'] for i in my_tasks]
        outputs: list[RequestOutput] = self.generate(
            input_data, sampling_param_inputs)
        for i in range(len(outputs)):
            response = []
            for j in outputs[i].outputs:
                response.append({
                    'text': truncate(j.text),
                    # 'token_ids': j.token_ids,
                    'cumulative_logprob': j.cumulative_logprob
                })

            full_response = my_tasks[i]
            full_response.update({
                'result': response
            })

            self.enqueue(
                obj=full_response,
                channel=my_tasks[i]['channel']  # The response channel.
            )

    def shutdown(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm
        gc.collect()
