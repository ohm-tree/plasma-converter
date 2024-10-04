import multiprocessing
from typing import Dict, List
import numpy as np


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

def policy_value_main(
        config: dict,
        run_name: str,
        policy_value_worker_id: int,
        gpu_set: List[int],
        master_queue: multiprocessing.Queue,
        policy_value_queue: multiprocessing.Queue,
        worker_queues: Dict[int, multiprocessing.Queue],
        global_policy_value_queue: multiprocessing.Queue,
):
    """
    Entry point for the PV worker process.

    Takes in contexts from the policy-value queue,
    and outputs suggestions for the lean game dicts.
    """

    import logging
    import os
    import queue

    # I live in src/workers/
    WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(WORKER_DIR)
    ROOT_DIR = os.path.dirname(SRC_DIR)

    # give myself a custom logging file.
    os.makedirs(f"{ROOT_DIR}/logs/{run_name}", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        f"logs/{run_name}/policy_value_worker_{policy_value_worker_id}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Starting policy-value worker {policy_value_worker_id}.")

    # Set up vllm stuff.
    import gc

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_set))

    # TODO: stuff all of the configs into a config file.
    llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
            seed=0,
            max_num_batched_tokens=8192,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=len(gpu_set))
    # llm = LLM(model="deepseek-ai/deepseek-math-7b-instruct",
    #           max_num_batched_tokens=8192,
    #           trust_remote_code=True,
    #           enforce_eager=True,
    #           tensor_parallel_size=len(gpu_set))

    sampling_params = SamplingParams(
        temperature=1,
        max_tokens=500,
        top_p=0.95,
        n=10,
        stop = ['\n']
    )

    logger.info("Policy-value worker initialized.")

    # bloodline = time.time()

    while True:
        # check my personal queue for blood.
        # found_blood = False
        kill = False
        while True:
            try:
                blood = policy_value_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if blood == 'kill':
                    # found_blood = False
                    kill = True
                    break
                # found_blood = True
                # bloodline = time.time()

        if kill:
            logger.fatal("Kill signal received. Dying.")
            break
        # if ((not found_blood) and time.time() < bloodline + 60 * 15):
        #     # If I haven't received a signal that other processes are
        #     # alive in the last 15 minutes, I should die.
        #     if not found_blood:
        #         logger.info("No blood found. Dying.")
        #         break

        my_tasks = []
        # tasks should take the form
        # {
        #   'mcts_worker_id': int, # The worker task id that generated this task.
        #   'task_id': int, # The specific completion task id of this task.
        #   'task_input': dict # The task to complete, which is a lean_game_dict.
        #   'type': dict
        # }

        try:
            new_task = global_policy_value_queue.get(timeout=30)
        except queue.Empty:
            pass
        else:
            assert new_task['type'] == 'context'
            my_tasks.append(new_task)
        while len(my_tasks) < config['batch_size']:
            try:
                new_task = global_policy_value_queue.get_nowait()
            except queue.Empty:
                break
            assert new_task['type'] == 'context'
            my_tasks.append(new_task)

        logger.info(f"PV Worker received {len(my_tasks)} tasks.")
        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            continue
        # We have tasks to complete.
        
        model_inputs = [prompt(task['task_input']) for task in my_tasks]

        model_outputs = llm.generate(
            model_inputs,
            sampling_params,
            use_tqdm=False,
        )

        for i in range(len(model_outputs)):
            options = model_outputs[i].outputs

            comments = np.array([option.text for option in options])
            policy = np.array([option.cumulative_logprob for option in options])
            unique_indices = [i==0 or comments[i]!=comments[i-1] for i in range(len(comments))]
            comments = comments[unique_indices]
            policy = policy[unique_indices]
            policy = np.exp(policy)
            policy /= policy.sum()

            res = {
                'comments': comments,
                'policy': policy,
                'value': 0.5
            }

            result = {
                'mcts_worker_id': my_tasks[i]['mcts_worker_id'],
                'task_id': my_tasks[i]['task_id'],
                'task_output': res,
                'type': 'policy_value'
            }
            logger.info(str(result))

            worker_queues[my_tasks[i]['mcts_worker_id']].put(result)

    # send a signal to the master queue that we are dead.
    master_queue.put({
        'name': 'policy_value',
        'worker_id': policy_value_worker_id,
        'type': 'dead'
    })

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
