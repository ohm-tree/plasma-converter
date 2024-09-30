"""
In this file, we will let a human play a game of Lean (using modal).
"""

import json
import logging
import multiprocessing
import os
import queue
import time

from src.games.leaner_lean_game import LeanGame, LeanGameState

# setup logging
logging.basicConfig(level=logging.INFO)


HOME_DIR = os.path.expanduser('~')
print("HOME_DIR", HOME_DIR)

with open(f"{HOME_DIR}/plasma-converter/datasets/minif2f.jsonl", 'r') as file:
    # Each line in the file is a separate JSON object
    data = [json.loads(line.strip()) for line in file.readlines()]

# comments = None
# with open("src/sample-data/comments.txt", 'r') as file:
#     comments = [line.strip() for line in file.readlines()]


def main(
    task_id: int,
    num_tasks: int,
    json_name: str,
    master_queue: multiprocessing.Queue,
    worker_queue: multiprocessing.Queue,
    completion_queue: multiprocessing.Queue,
    lean_queue: multiprocessing.Queue,
    context_queue: multiprocessing.Queue
):

    # give myself a custom logging file.
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        f"logs/distributed_leaner_game_test_cpu_worker_{task_id}.log")
    logger.addHandler(fh)
    logger.info(f"Starting distributed_leaner_game_test_cpu_worker {task_id}.")

    for current_problem in range(task_id, len(data), num_tasks):
        logger.info(f"Worker {task_id} working on problem {current_problem}")
        problem = data[current_problem]
        informal_prefix = problem['informal_prefix']
        formal_statement = problem['formal_statement']
        PROBLEM_STATEMENT = informal_prefix + formal_statement
        tactic_state = problem['goal']

        game: LeanGame = LeanGame(
            # comment_seeds=comments,
            num_comment_seeds=6,
            max_depth=20
        )
        state: LeanGameState = game.start_state(
            problem=PROBLEM_STATEMENT,
            tactic_state=tactic_state
        )

        context_queue.put(
            {
                'mcts_worker_id': task_id,
                # In this test, we will only have one context task ever and the cpu workers will spin on the outputs.
                'task_id': 0,
                'task_input': state.pre_comments(),
                'type': 'context'
            }
        )

        time_to_context = -time.time()
        context_output = None
        while context_output is None:
            try:
                context_output = worker_queue.get_nowait()
            except queue.Empty:
                context_output = None
                pass

        time_to_context += time.time()
        logger.info(f"Time to context: {time_to_context}")
        assert context_output['type'] == 'policy_value'
        assert context_output['task_id'] == 0
        state.post_comments(context_output['task_output']['comments'])
        logger.info(
            f"Received policy: {context_output['task_output']['policy']}")
        logger.info("Received value: {context_output['task_output']['value']}")
        # first, we need to comment the state right away.

        while not game.is_terminal(state):
            logger.info(state.human_printout())
            action = 0
            state = game.next_state(state, action)

            input_data = state.pre_LLM_rollout()

            # tasks should take the form
            # {
            #   'worker_id': int, # The worker task id that generated this task.
            #   'completion_task_id': int, # The specific completion task id of this task.
            #   'task': str # The task to complete, a string prompt.
            #   'type': str # Should be 'completion'
            # }
            completion_queue.put({
                'mcts_worker_id': task_id,
                # In this test, we will only have one completion task ever and the cpu workers will spin on the outputs.
                'completion_task_id': 0,
                'task': input_data,
                'type': 'completion'
            })
            time_to_completion = -time.time()
            completion_output = None
            while completion_output is None:
                try:
                    completion_output = worker_queue.get_nowait()
                except queue.Empty:
                    completion_output = None
                    pass

            time_to_completion += time.time()
            logger.info(f"Time to completion: {time_to_completion}")
            assert completion_output['type'] == 'completion'
            assert completion_output['completion_task_id'] == 0

            output = completion_output['output'] + "\n"
            state.post_LLM_rollout(output)
            lean4_input = state.pre_process()
            # tasks should take the form
            # {
            #   'worker_id': int, # The worker task id that generated this task.
            #   'lean_task_id': int, # The specific lean task id of this task.
            #   'task': str # The task to complete, a string prompt.
            #   'type': str # Should be 'lean'
            # }

            lean_queue.put({
                'mcts_worker_id': task_id,
                # In this test, we will only have one lean task ever and the cpu workers will spin on the outputs.
                'lean_task_id': 0,
                'task': lean4_input,
                'type': 'lean'
            })

            time_to_lean = -time.time()
            lean_output = None
            while lean_output is None:
                try:
                    lean_output = worker_queue.get_nowait()
                except queue.Empty:
                    lean_output = None
                    pass

            time_to_lean += time.time()
            logger.info(f"Time to lean: {time_to_lean}")
            assert lean_output['type'] == 'lean'
            assert lean_output['lean_task_id'] == 0
            state.post_process(lean_output['result'])

            context_queue.put(
                {
                    'mcts_worker_id': task_id,
                    # In this test, we will only have one context task ever and the cpu workers will spin on the outputs.
                    'task_id': 0,
                    'task_input': state.pre_comments(),
                    'type': 'context'
                }
            )

            time_to_context = -time.time()
            context_output = None
            while context_output is None:
                try:
                    context_output = worker_queue.get_nowait()
                except queue.Empty:
                    context_output = None
                    pass

            time_to_context += time.time()
            logger.info(f"Time to context: {time_to_context}")
            assert context_output['type'] == 'policy_value'
            assert context_output['task_id'] == 0
            state.post_comments(context_output['task_output']['comments'])
            logger.info(
                f"Received policy: {context_output['task_output']['policy']}")
            logger.info(
                f"Received value: {context_output['task_output']['value']}")

        # save the human printout to a file
        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/{problem['name']}.txt", 'w') as file:
            file.write(state.human_printout())

        logger.info(f"Finished problem {problem['name']} result: {state.win}")
    # tell the master queue that we are done with all tasks.
    master_queue.put(
        {
            'mcts_worker_id': task_id,
            'task_id': 0,
            'type': 'done'
        }
    )
