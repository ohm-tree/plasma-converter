
import multiprocessing
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from vllm import RequestOutput

from src.workers.llm_worker import LLMWorker
from src.workers.types import FastPolicyValueWorkerType, PolicyValueTaskType
from src.workers.worker import TaskType, WorkerIdentifer, WorkerTask


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


class FastPolicyValueWorker(LLMWorker):
    def __init__(self,
                 global_config: dict,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 gpu_set: List[int],
                 queues: Dict[Union[TaskType, WorkerIdentifer], multiprocessing.Queue],
                 ):
        super().__init__(
            worker_id=WorkerIdentifer(
                FastPolicyValueWorkerType, task_id),
            queues=queues,
            run_name=run_name,
            gpu_set=gpu_set,
            config=config,
            # run_locally=global_config['run_locally'],
            # LLM_kwargs=config['model'],
            # sampling_kwargs=config['sampling']
        )
        self.config = config
        self.num_comments = config['num_comments']

        assert config['num_comments'] == config['sampling']['n'] + 1

    def loop(self):
        my_tasks: Iterator[WorkerTask] = self.spin_deque_task(
            channel=PolicyValueTaskType,
            timeout=30,
            batch_size=self.config['batch_size'],
        )

        self.logger.info(f"Received {len(my_tasks)} tasks.")
        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            return
        # We have tasks to complete.
        model_inputs = [prompt(task.task) for task in my_tasks]

        model_outputs: List[RequestOutput] = self.generate(
            model_inputs
        )

        for i in range(len(model_outputs)):
            options = model_outputs[i].outputs

            comments = np.array([option.text for option in options])
            policy = np.array(
                [option.cumulative_logprob for option in options])
            # unique_indices = [i == 0 or comments[i] != comments[i-1]
            #                   for i in range(len(comments))]
            # comments = comments[unique_indices]
            # policy = policy[unique_indices]
            policy = np.exp(policy)
            policy /= policy.sum()

            comments = [''] + [option.text for option in options]
            assert len(comments) == self.num_comments
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
