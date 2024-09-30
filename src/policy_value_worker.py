import json
import multiprocessing
import os
import queue
import time
from typing import Dict

from vllm import LLM, SamplingParams


def suggest_comments_prompts(lean_game_dict: Dict, num: int = 5) -> str:
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
Please summarize the current tactic state of the proof. Then, please discuss whether or not the proof is on the right track; are we making progress? Are we stuck? Are we going in the wrong direction or repeating ourselves?
"""

    # We should call the LLM now.

    res_2 = f"""Now, please suggest {num} ideas to complete the proof. Please delimit each idea with <IDEA></IDEA> tags.
Finally, rate the likelihood that this proof will succeed on a scale of 1 (very unlikely) to 10 (very likely). Please delimit this rating with <RATING></RATING> tags.
"""
    return [res, res_2]


def main(
        task_id: int,
        num_tasks: int,
        json_name: str,
        master_queue: multiprocessing.Queue,
        worker_queues: Dict[int, multiprocessing.Queue],
        policy_value_queue: multiprocessing.Queue,
        policy_value_batch_size: int,
):
    """
    Entry point for the lean worker process.
    """

    # TODO: stuff all of the configs into a config file.
    llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
              max_num_batched_tokens=8192,
              trust_remote_code=True)

    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        top_k=1,
        top_p=1.0
    )

    while True:
        # check for kill signals from the master queue.
        try:
            kill_signal = master_queue.get_nowait()
            print(f"Worker {task_id} received kill signal: {kill_signal}")
            if kill_signal == "kill":
                break
        except queue.Empty:
            pass

        my_tasks = []
        # tasks should take the form
        # {
        #   'worker_id': int, # The worker task id that generated this task.
        #   'task_id': int, # The specific completion task id of this task.
        #   'task_input': str # The task to complete, a string prompt.
        #   'task': dict
        # }
        while len(my_tasks) < policy_value_batch_size:
            try:
                new_task = policy_value_queue.get_nowait()
            except queue.Empty:
                break
            assert new_task['task'] == 'policy_value'
            my_tasks.append(new_task)

        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            time.sleep(1)
        else:
            # We have tasks to complete.
            input_data = [
                my_tasks[i]['task_input']
                for i in range(len(my_tasks))
            ]
            outputs = llm.generate(
                input_data,
                sampling_params=sampling_params
            )
            # TODO: generate

            for i in range(len(outputs)):
                result = {
                    'task_id': my_tasks[i]['task_id'],
                    'output': outputs[i].outputs[0].text
                }

                worker_queues[my_tasks[i]['worker_task_id']].put(result)
