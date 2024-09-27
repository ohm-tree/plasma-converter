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

comments = None
with open("src/sample-data/comments.txt", 'r') as file:
    comments = [line.strip() for line in file.readlines()]


def main(worker_queue: multiprocessing.Queue,
         completion_queue: multiprocessing.Queue,
         lean_queue: multiprocessing.Queue,
         task_id: int,
         num_tasks: int,
         json_name: str
         ):

    # give myself a custom logging file.
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename=f"logs/worker_{task_id}.log", level=logging.INFO)

    for current_problem in range(task_id, len(data), num_tasks):
        print(f"Worker {task_id} working on problem {current_problem}")
        problem = data[current_problem]
        informal_prefix = problem['informal_prefix']
        formal_statement = problem['formal_statement']
        PROBLEM_STATEMENT = informal_prefix + formal_statement
        tactic_state = problem['goal']

        game: LeanGame = LeanGame(
            comment_seeds=comments,
            max_depth=20
        )
        state: LeanGameState = game.start_state(
            problem=PROBLEM_STATEMENT,
            tactic_state=tactic_state
        )

        while not game.is_terminal(state):
            logging.info(state.human_printout())
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
                'worker_id': task_id,
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
            print(f"Time to completion: {time_to_completion}")
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
                'worker_id': task_id,
                # In this test, we will only have one lean task ever and the cpu workers will spin on the outputs.
                'lean_task_id': 0,
                'task': lean4_input,
                'type': 'lean'
            })

            time_to_lean = -time.time()
            lean4_output = None
            while lean4_output is None:
                try:
                    lean4_output = worker_queue.get_nowait()
                except queue.Empty:
                    lean_4_output = None
                    pass
            time_to_lean += time.time()
            print(f"Time to lean: {time_to_lean}")
            assert lean4_output['type'] == 'lean'
            assert lean4_output['lean_task_id'] == 0
            state.post_process(lean4_output)
        # save the human printout to a file
        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/{problem['name']}.txt", 'w') as file:
            file.write(state.human_printout())

        logging.info(f"Finished problem {problem['name']} result: {state.win}")
