# Set up vllm stuff.
import gc
import time
from secrets import token_bytes

# remote queries to openai
from openai import OpenAI
from ray import worker
from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from src.workers.performance_logger import PerformanceLogger
from src.workers.worker import *


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

        self.llm = LLM(
            **LLM_kwargs
        )
        self.gpu_set = gpu_set
        self.use_tqdm = use_tqdm
        self.performance_logger = PerformanceLogger()

        self.chat_mode = config.get('chat', False)  # Get chat flag from config

        if self.chat_mode:
            self.channel = "chat"
        else:
            self.channel = "completion"

    def generate(self, input_data: list[str | list[dict]],
                 sampling_params: list[SamplingParams],
                 ) -> list[RequestOutput]:
        if self.chat_mode:
            return self.llm.chat(
                messages=input_data,
                sampling_params=sampling_params,
                use_tqdm=self.use_tqdm
            )
        else:
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
        # self.logger.info(
        #     f"Received {len(my_tasks)} tasks.")
        start_time = time.time()

        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            return
        # We have tasks to complete.
        if self.chat_mode:
            input_data = [i['messages']
                          for i in my_tasks]  # Get messages for chat mode
        else:
            input_data = [i['prompt']
                          for i in my_tasks]  # Original prompt handling

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

        # sampling_param_inputs = [i['sampling_params'] for i in my_tasks]
        outputs: list[RequestOutput] = self.generate(
            input_data, sampling_param_inputs)

        total_waiting_time = 0
        for i in range(len(outputs)):
            response = []
            for j in outputs[i].outputs:
                response.append({
                    'text': j.text,
                    'token_ids': j.token_ids,
                    'cumulative_logprob': j.cumulative_logprob
                })

            full_response = my_tasks[i]
            full_response.update({
                'result': response
            })

            total_waiting_time += start_time - full_response['enqueue_time']

            self.enqueue(
                obj=full_response,
                channel=my_tasks[i]['channel']  # The response channel.
            )
        end_time = time.time()
        self.performance_logger.log_query(
            latency=end_time - start_time,
            total_waiting_time=total_waiting_time,
            quantity=len(my_tasks)
        )
        self.performance_logger.occasional_log(self.logger)

    def shutdown(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor
        del self.llm
        gc.collect()
