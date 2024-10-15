
import multiprocessing
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from vllm import RequestOutput

from src.workers.llm_worker import LLMWorker
from src.workers.types import FastPolicyValueWorkerType, PolicyValueTaskType
from src.workers.worker import TaskType, WorkerIdentifer, WorkerTask
from pprint import pprint


def prompt(lean_game_dict: Dict) -> str:
    """
    Generate a prompt for the policy-value worker to suggest comments.
    """
    # print("lean_game_dict")
    # pprint(lean_game_dict)
    prompt = 'Complete the following Lean 4 code with explanatory comments.' + \
            '```lean\n' + lean_game_dict['header'] + lean_game_dict['problem'] + \
            lean_game_dict['old_code'] + \
            lean_game_dict['valid_code'] + \
            "\n  /-- The tactic state is:\n" + '\n'.join(['  ' + line for line in lean_game_dict['tactic_state'].strip().splitlines()]) + "\n  -/\n" + \
            "  --" 
    # print("prompt")
    # print(prompt)
    return prompt


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
        )
        self.config = config
        # assert global_config['branching_factor'] == config['sampling']['n'] 

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

        # self.logger.info(f"Generated {len(model_inputs)} prompts.")
        # self.logger.info(model_inputs)

        model_outputs: List[RequestOutput] = self.generate(
            model_inputs
        )

        # self.logger.info(f"Generated {len(model_outputs)} outputs.")
        # self.logger.info(model_outputs)

        for i in range(len(model_outputs)):
            options = model_outputs[i].outputs

            snippets = [option.text for option in options]

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
            snippets = ["  --" + truncate(snippet) for snippet in snippets]

            # print("snippets", snippets)

            policy = np.array(
                [option.cumulative_logprob for option in options])
            # print("log probs", policy)
            # unique_indices = [i == 0 or comments[i] != comments[i-1]
            #                   for i in range(len(comments))]
            # comments = comments[unique_indices]
            # policy = policy[unique_indices]
            policy = np.exp(policy/3)
            policy /= policy.sum()
            # print("policy", policy)

            # snippets = [''] + [option.text for option in options]
            res = {
                'moves': snippets,
                'policy': policy,
                'value': 0.5
            }

            self.logger.info(str(res))

            self.enqueue_response(
                response=res,
                task=my_tasks[i]
            )
