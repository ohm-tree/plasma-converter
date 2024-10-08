
from typing import Dict, List

import numpy as np
from vllm import RequestOutput

from src.workers.llm_worker import LLMWorker
from src.workers.worker import *


def prompt(lean_game_dict: Dict) -> str:
    """
    Generate a prompt for the policy-value worker to suggest comments.
    """
    res = r'''Complete the following Lean 4 code with short explanatory comments:

```lean4
'''

    res += lean_game_dict['header'] + \
        lean_game_dict['problem'] + lean_game_dict['old_code']

    return res


PolicyValueTaskType = TaskType("policy_value")
FastPolicyValueWorkerType = WorkerType(
    "fast_policy_value", [PolicyValueTaskType])


class FastPolicyValueWorker(LLMWorker):
    def __init__(self,
                 config: dict,
                 run_name: str,
                 fast_pv_worker_id: int,
                 gpu_set: List[int],
                 queues: Dict[Union[TaskType, WorkerIdentifer]],
                 ):
        super().__init__(
            worker_id=WorkerIdentifer(
                FastPolicyValueWorkerType, fast_pv_worker_id),
            queues=queues,
            run_name=run_name,
            gpu_set=gpu_set,
            LLM_kwargs=None,  # Default to the LLM constructor.
            sampling_kwargs={
                'temperature': 1,
                'max_tokens': 500,
                'top_p': 0.95,
                'n': 10,
                'stop': ['\n']
            }
        )
        self.config = config

    def loop(self):
        my_tasks: Iterable[WorkerTask] = self.spin_deque_task(
            task_type=PolicyValueTaskType,
            timeout=30,
            max_tasks=self.config['batch_size'],
        )

        self.logger.info(f"Received {len(my_tasks)} tasks.")
        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            return
        # We have tasks to complete.
        model_inputs = [prompt(task['task_input']) for task in my_tasks]

        model_outputs: List[RequestOutput] = self.generate(
            model_inputs
        )

        for i in range(len(model_outputs)):
            options = model_outputs[i].outputs

            comments = np.array([option.text for option in options])
            policy = np.array(
                [option.cumulative_logprob for option in options])
            unique_indices = [i == 0 or comments[i] != comments[i-1]
                              for i in range(len(comments))]
            comments = comments[unique_indices]
            policy = policy[unique_indices]
            policy = np.exp(policy)
            policy /= policy.sum()

            res = {
                'comments': comments,
                'policy': policy,
                'value': 0.5
            }

            self.logger.info(str(res))

            self.enqueue_response(
                response=res,
                task=my_tasks[i]
            )
