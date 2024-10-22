import json
import logging
import multiprocessing
import os
import queue
import time
from typing import dict, list

from vllm import RequestOutput

from src.workers.llm_worker import LLMWorker
from src.workers.types import (
    ContextWorkerType,
    PolicyValuePostProcessTaskType,
    PolicyValueTaskType,
    PolicyValueWorkerType,
)
from src.workers.worker import *


def construct_context(lean_game_dict: dict) -> str:
    """
    Generate a prompt for the policy-value worker to suggest comments.
    """
    res = """This is a partial Lean 4 proof.
```lean4
"""
    res += lean_game_dict['header'] + \
        lean_game_dict['problem'] + lean_game_dict['old_code']
    res += """
```
Here is the tactic state at this point:
```lean4
"""
    res += lean_game_dict['tactic_state']
    res += f"""
```
Please summarize what we have proven so far.
Please summarize the current tactic state of the proof.
Then, please discuss whether or not the proof is on the right track. Are we proving useful lemmas? Are we using the right tactics? Are we missing any key insights?
"""
    return res


def policy_value_suggest_comments(lean_game_dict: dict, discussion_context: str, num: int = 5) -> str:
    # We should call the LLM now.

    res = discussion_context + f"""Here is the tactic state at this point:
```lean4
"""
    res += lean_game_dict['tactic_state']
    res += f"""
```
Then, please suggest {num} ideas to complete the proof.
Please delimit each idea with <IDEA></IDEA> tags.

Rate the likelihood that this proof will succeed on a scale of 1 (very unlikely) to 10 (very likely).
Please delimit this rating with a single number inside <RATING></RATING> tags.

<IDEA>"""
    return res


def parse_policy_value_output(output: str, logger: logging.Logger,
                              num: int = 5) -> dict:
    """
    Parse the output of the policy-value worker into a dict.

    Parameters:
    ----------
    output: str
        The output of the policy-value worker.
    logger: logging.Logger
        The logger to log any warnings.
    num: int
        The number of comments that the LLM should have generated.
        The output will contain num + 1 comments, where the first comment is the empty string.

    Returns:
    -------
    res: dict
        A dictionary containing the rating, comments, policy, and value.
    """
    res = {}

    # We truncated the first <IDEA> for prompting purposes...
    output = "<IDEA>" + output

    try:
        rating_output = output.split("<RATING>")[1]
        res['rating'] = int(rating_output.split("</RATING>")[0])
    except:
        logger.warning(f"Rating output is not a number.")
        res['rating'] = 5  # default to 5 if the rating is not a number.

    idea_outputs = output.split("<IDEA>")
    res['comments'] = [""]
    for i in range(1, min(len(idea_outputs), num + 1)):
        idea = idea_outputs[i].split("</IDEA>")[0]
        res['comments'].append(idea)

    if len(res['comments']) < num + 1:
        # Default to empty strings if there are not enough comments.
        logger.warning(
            f"Number of comments is less than expected: {len(res['comments']) - 1}")
        res['comments'] += [""] * (num + 1 - len(res['comments']))

    # pre-pend the empty comment.

    # TODO: for now, we will just return a uniform distribution over the ideas.
    res['policy'] = [1.0 / (num + 1) for _ in range(num + 1)]
    res['value'] = res['rating'] / 10.0

    return res


class ContextWorker(LLMWorker):

    def __init__(self,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 gpu_set: list[int],
                 queues: dict[str, multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            worker_id=WorkerIdentifer(
                ContextWorkerType, task_id),
            queues=queues,
            run_name=run_name,
            gpu_set=gpu_set,
            config=config
            # run_locally=config['run_locally'],
            # LLM_kwargs=config['model'],
            # sampling_kwargs=config['sampling']
        )
        self.config = config

    def loop(self):
        my_tasks: list[WorkerTask] = self.spin_deque_task(
            channel=PolicyValueTaskType,
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
            construct_context(i.task)
            for i in my_tasks
        ]
        outputs: list[RequestOutput] = self.generate(
            input_data
        )

        for i in range(len(outputs)):
            output = outputs[i].outputs[0].text
            # self.logger.info(output)
            task: WorkerTask = my_tasks[i]
            wrapped_response = WorkerResponse(
                head_id=task.head_id,
                tail_id=self.worker_id,
                task_id=TaskIdentifier(
                    task_idx=task.task_id.task_idx,
                    task_type=PolicyValuePostProcessTaskType
                ),
                task=task.task,
                response=output
            )
            # self.logger.info(wrapped_response)
            self.enqueue(
                obj=wrapped_response,
                where=PolicyValuePostProcessTaskType
            )


class PolicyValueWorker(LLMWorker):
    def __init__(self,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 gpu_set: list[int],
                 queues: dict[str, multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            worker_id=WorkerIdentifer(
                PolicyValueWorkerType, task_id),
            queues=queues,
            run_name=run_name,
            gpu_set=gpu_set,
            config=config,
            # run_locally=config['run_locally'],
            # LLM_kwargs=config['model'],
            # sampling_kwargs=config['sampling']
        )
        self.config = config
        self.num = config['num_comments']

    def loop(self):
        my_tasks: list[WorkerResponse] = self.spin_deque_task(
            channel=PolicyValuePostProcessTaskType,
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
            policy_value_suggest_comments(
                i.task,
                i.response,
                num=self.num
            )
            for i in my_tasks
        ]
        outputs: list[RequestOutput] = self.generate(
            input_data
        )

        for i in range(len(outputs)):
            output = outputs[i].outputs[0].text
            # self.logger.info(output)
            res = parse_policy_value_output(
                output, self.logger, num=self.num)

            self.enqueue_response(
                response=res,
                task=WorkerTask(
                    head_id=my_tasks[i].head_id,
                    task_id=TaskIdentifier(
                        task_idx=my_tasks[i].task_id.task_idx,
                        task_type=PolicyValueTaskType
                    ),
                    task=my_tasks[i].task
                )
            )
