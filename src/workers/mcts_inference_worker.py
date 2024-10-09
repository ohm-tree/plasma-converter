"""
In this file, we will let a human play a game of Lean.
"""

import multiprocessing
import json
import logging
import os

import numpy as np

from src.games.lean_game import LeanGame, LeanGameState
from src.train.self_play import self_play


def main(
    config: dict,
    run_name: str,
    task_id: int,
    master_queue: multiprocessing.Queue,
    worker_queue: multiprocessing.Queue,
    global_completion_queue: multiprocessing.Queue,
    global_lean_queue: multiprocessing.Queue,
    global_context_queue: multiprocessing.Queue
):
    # I live in src/workers/
    WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(WORKER_DIR)
    ROOT_DIR = os.path.dirname(SRC_DIR)

    with open(config['data_dir'], 'r') as file:
        data = [
            json.loads(line.strip()) 
            for line in file.readlines() 
            if json.loads(line.strip()).get('split') == config['split']
        ]

    # give myself a custom logging file.
    os.makedirs(f"{ROOT_DIR}/logs/{run_name}", exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(
        f"logs/{run_name}/mcts_inference_worker_{task_id}.log")
    logger.addHandler(fh)
    logger.info(f"Starting mcts_inference_worker {task_id}.")

    for current_problem in range(task_id, len(data), config['worker']['num_procs']):
        logger.info(f"Worker {task_id} working on problem {current_problem}")
        problem = data[current_problem]
        informal_prefix = problem['informal_prefix']
        formal_statement = problem['formal_statement']
        PROBLEM_STATEMENT = informal_prefix + formal_statement
        tactic_state = problem['goal']

        game: LeanGame = LeanGame(
            # comment_seeds=comments,
            num_comment_seeds=config['policy_value']['branching_factor'],
            max_depth=config['max_depth'],
        )
        state: LeanGameState = game.start_state(
            problem=PROBLEM_STATEMENT,
            tactic_state=tactic_state
        )

        states, distributions, rewards = self_play(
            worker_id=task_id,
            state=state,
            game=game,
            num_iters=1000,
            logger=logger,
            worker_queue=worker_queue,
            global_completion_queue=global_completion_queue,
            global_context_queue=global_context_queue,
            global_lean_queue=global_lean_queue
        )

        game_data_path = f"data/{run_name}/games/{task_id}"
        os.makedirs(game_data_path, exist_ok=True)

        LeanGameState.saves(states, os.path.join(
            game_data_path, f"{problem['name']}_states.npy"))

        with open(os.path.join(game_data_path, f"{problem['name']}_distributions.npy"), "wb") as file:
            # Don't allow pickle, I want this to be a numpy array for sure.
            np.save(file, distributions, allow_pickle=False)
        with open(os.path.join(game_data_path, f"{problem['name']}_outcomes.npy"), "wb") as file:
            # Don't allow pickle, I want this to be a numpy array for sure.
            np.save(file, rewards, allow_pickle=False)

        # save the human printout to a file
        os.makedirs(f"outputs/{run_name}/", exist_ok=True)
        with open(f"outputs/{run_name}/{problem['name']}.txt", 'w') as file:
            file.write(states[-1].human_printout())

        logger.info(f"Finished problem {problem['name']} result: {rewards[-1]}")

    # tell the master queue that we are done with all tasks.
    master_queue.put(
        {
            'mcts_worker_id': task_id,
            'task_id': 0,
            'type': 'done'
        }
    )
