"""
Entry point for the completion worker process.
"""

import multiprocessing
from typing import Dict, List

from vllm import RequestOutput

from src.workers.llm_worker import LLMWorker
from src.workers.types import CompletionTaskType, CompletionWorkerType
from src.workers.worker import *


class CompletionWorker(LLMWorker):
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
                CompletionWorkerType, task_id),
            queues=queues,
            run_name=run_name,
            gpu_set=gpu_set,
            LLM_kwargs=config['model'],
            sampling_kwargs=config['sampling']
        )
        self.config = config

    def loop(self):
        my_tasks: Iterable[WorkerTask] = self.spin_deque_task(
            channel=CompletionTaskType,
            timeout=30,
            batch_size=self.config['batch_size'],
        )
        self.logger.info(
            f"Received {len(my_tasks)} tasks.")

        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            return
        # We have tasks to complete.
        input_data = [
            i.task for i in my_tasks
        ]
        outputs: List[RequestOutput] = self.generate(
            input_data,
            sampling_params=self.sampling_params
        )
        for i in range(len(outputs)):
            output = outputs[i].outputs[0].text
            self.logger.info(output)
            self.enqueue_response(
                response=output,
                task=my_tasks[i]
            )
