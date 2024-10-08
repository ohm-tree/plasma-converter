import json
import logging
import multiprocessing
import os
import queue
import time
from typing import Dict, List


def construct_context(lean_game_dict: Dict) -> str:
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


def policy_value_suggest_comments(lean_game_dict: Dict, discussion_context: str, num: int = 5) -> str:
    # We should call the LLM now.

    res = discussion_context + f"""Here is the tactic state at this point:
```lean4
"""
    res += lean_game_dict['tactic_state']
    res += f"""
```
Then, please suggest 5 ideas to complete the proof.
Please delimit each idea with <IDEA></IDEA> tags.

Rate the likelihood that this proof will succeed on a scale of 1 (very unlikely) to 10 (very likely).
Please delimit this rating with a single number inside <RATING></RATING> tags.

<IDEA>"""
    return res


def parse_policy_value_output(output: str, logger: logging.Logger,
                              num: int = 5) -> Dict:
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
    res: Dict
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


def context_main(
        run_name: str,
        context_worker_id: int,
        num_context_workers: int,
        json_name: str,
        gpu_set: List[int],
        master_queue: multiprocessing.Queue,
        context_queue: multiprocessing.Queue,
        global_context_queue: multiprocessing.Queue,
        global_policy_value_queue: multiprocessing.Queue,
        context_batch_size: int,
):
    """
    Entry point for the context worker process.

    Takes in lean game dicts from the context queue,
    and outputs a context for the policy-value worker to suggest comments.
    """

    # I live in src/workers/
    WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(WORKER_DIR)
    ROOT_DIR = os.path.dirname(SRC_DIR)

    # give myself a custom logging file.
    os.makedirs(f"{ROOT_DIR}/logs/{run_name}", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        f"logs/{run_name}/context_worker_{context_worker_id}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f"Starting context worker {context_worker_id}.")

    # Set up vllm stuff.
    import gc

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_set))

    # TODO: stuff all of the configs into a config file.
    # llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
    #           max_num_batched_tokens=8192,
    #           trust_remote_code=True,
    #           dtype="float16",
    #           tensor_parallel_size=len(gpu_set))
    llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
              max_num_batched_tokens=8192,
              trust_remote_code=True,
              tensor_parallel_size=len(gpu_set))

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
        top_k=1,
        top_p=1.0
    )

    logger.info("Context worker initialized.")

    # bloodline = time.time()

    while True:
        # check my personal queue for blood.
        # found_blood = False
        kill = False
        while True:
            try:
                blood = context_queue.get_nowait()
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
            logger.info("Kill signal received. Dying.")
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
            new_task = global_context_queue.get(timeout=30)
        except queue.Empty:
            pass
        else:
            assert new_task['type'] == 'context'
            my_tasks.append(new_task)

        while len(my_tasks) < context_batch_size:
            try:
                new_task = global_context_queue.get_nowait()
            except queue.Empty:
                break
            assert new_task['type'] == 'context'
            my_tasks.append(new_task)

        logger.info(f"Context Worker Received {len(my_tasks)} tasks.")
        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            continue
        # We have tasks to complete.
        input_data = [
            construct_context(my_tasks[i]['task_input'])
            for i in range(len(my_tasks))
        ]
        outputs = llm.generate(
            input_data,
            sampling_params=sampling_params
        )

        for i in range(len(outputs)):
            result = {
                'mcts_worker_id': my_tasks[i]['mcts_worker_id'],
                'task_id': my_tasks[i]['task_id'],
                'task_input': my_tasks[i]['task_input'],
                'task_context': outputs[i].outputs[0].text,
                'type': 'policy_value'
            }
            logger.info(str(result))
            global_policy_value_queue.put(result)
    # send a signal to the master queue that we are dead.
    master_queue.put({
        'name': 'context',
        'worker_id': context_worker_id,
        'type': 'dead'
    })

    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()


def policy_value_main(
        run_name: str,
        policy_value_worker_id: int,
        num_policy_value_workers: int,
        json_name: str,
        gpu_set: List[int],
        master_queue: multiprocessing.Queue,
        policy_value_queue: multiprocessing.Queue,
        worker_queues: Dict[int, multiprocessing.Queue],
        global_policy_value_queue: multiprocessing.Queue,
        policy_value_batch_size: int,
):
    """
    Entry point for the PV worker process.

    Takes in contexts from the policy-value queue,
    and outputs suggestions for the lean game dicts.
    """

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
    # llm = LLM(model="deepseek-ai/deepseek-math-7b-instruct",
    #           max_num_batched_tokens=8192,
    #           trust_remote_code=True,
    #           dtype="float16",
    #           tensor_parallel_size=len(gpu_set))
    llm = LLM(model="deepseek-ai/deepseek-math-7b-instruct",
              max_num_batched_tokens=8192,
              trust_remote_code=True,
              tensor_parallel_size=len(gpu_set))

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.0,
        top_k=1,
        top_p=1.0
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
        #   'task_input': str # The task to complete, a string prompt.
        #   'type': dict
        # }
        try:
            new_task = global_policy_value_queue.get(timeout=30)
        except queue.Empty:
            pass
        else:
            assert new_task['type'] == 'policy_value'
            my_tasks.append(new_task)
        while len(my_tasks) < policy_value_batch_size:
            try:
                new_task = global_policy_value_queue.get_nowait()
            except queue.Empty:
                break
            assert new_task['type'] == 'policy_value'
            my_tasks.append(new_task)

        logger.info(f"PV Worker received {len(my_tasks)} tasks.")
        if len(my_tasks) == 0:
            # Spinlock, disappointing, but there's nothing to do.
            continue
        # We have tasks to complete.
        input_data = [
            policy_value_suggest_comments(
                my_tasks[i]['task_input'],
                my_tasks[i]['task_context']
            )
            for i in range(len(my_tasks))
        ]
        outputs = llm.generate(
            input_data,
            sampling_params=sampling_params
        )

        for i in range(len(outputs)):
            res = parse_policy_value_output(
                outputs[i].outputs[0].text, logger)

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
