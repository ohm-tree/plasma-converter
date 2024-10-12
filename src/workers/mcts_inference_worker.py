"""
In this file, we will let a human play a game of Lean.
"""

import json
import logging
import multiprocessing
import os
import queue
import time
from typing import Dict, List, Optional, Union

import numpy as np

from src.games.lean_game import MetaLeanGameMove, MetaLeanGameState
from src.train.self_play import self_play
from src.workers.types import (
    CompletionTaskType,
    LeanTaskType,
    MCTSWorkerType,
    PolicyValueTaskType,
)
from src.workers.worker import TaskType, Worker, WorkerIdentifer, WorkerType


class MCTSWorker(Worker):
    def __init__(self,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: Dict[Union[TaskType, WorkerIdentifer], multiprocessing.Queue],
                 ):
        super().__init__(
            worker_id=WorkerIdentifer(
                MCTSWorkerType, task_id),
            queues=queues,
            run_name=run_name,
            poison_scream=False
        )

        self.config = config

        self.load_problems()

        self.game_data_path = f"data/{run_name}/games/{task_id}/"
        os.makedirs(self.game_data_path, exist_ok=True)
        self.output_path = f"outputs/{run_name}/"
        os.makedirs(self.output_path, exist_ok=True)

    def load_problems(self):
        # I live in src/workers/
        WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
        SRC_DIR = os.path.dirname(WORKER_DIR)
        ROOT_DIR = os.path.dirname(SRC_DIR)

        with open(self.config['data_dir'], 'r') as file:
            self.data = [
                json.loads(line.strip())
                for line in file.readlines()
                if json.loads(line.strip()).get('split') == self.config['split']
            ]

    def run(self):
        for current_problem in range(self.worker_idx, len(self.data), self.config['worker']['num_procs']):
            self.logger.info(
                f"Working on problem {current_problem}")
            problem = self.data[current_problem]
            informal_prefix = problem['informal_prefix']
            formal_statement = problem['formal_statement']
            PROBLEM_STATEMENT = informal_prefix + formal_statement
            tactic_state = problem['goal']

            game: LeanGame = LeanGame(
                # comment_seeds=comments,
                num_comment_seeds=6,
                max_depth=config['max_depth']
            )
            state: LeanGameState = game.start_state(
                problem=PROBLEM_STATEMENT,
                tactic_state=tactic_state
            )

            states: List[LeanGameState]

            states, distributions, rewards = self_play(
                self,
                state=state,
                game=game,
                num_iters=1000,
            )

            LeanGameState.saves(states, os.path.join(
                self.game_data_path, f"{problem['name']}_states.npy"))

            with open(os.path.join(self.game_data_path, f"{problem['name']}_distributions.npy"), "wb") as file:
                # Don't allow pickle, I want this to be a numpy array for sure.
                np.save(file, distributions, allow_pickle=False)
            with open(os.path.join(self.game_data_path, f"{problem['name']}_outcomes.npy"), "wb") as file:
                # Don't allow pickle, I want this to be a numpy array for sure.
                np.save(file, rewards, allow_pickle=False)

            # save the human printout to a file
            with open(os.path.join(self.output_path, f"{problem['name']}.txt"), 'w') as file:
                for i, state in enumerate(states):
                    file.write(state.human_printout())

            self.logger.info(
                f"Finished problem {problem['name']} result: {rewards[-1]}")

    def loop(self):
        pass
