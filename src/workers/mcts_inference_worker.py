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
from src.workers import *
from src.utils import upload_to_gcs


class MCTSWorker(Worker):
    def __init__(self,
                 global_config: dict,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: Dict[Union[TaskType, WorkerIdentifer], multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            worker_id=WorkerIdentifer(
                MCTSWorkerType, task_id),
            queues=queues,
            run_name=run_name,
            poison_scream=False
        )

        self.config = config
        self.global_config = global_config
        self.num_iters = self.config['num_iters']
        self.max_actions = self.config['max_actions']

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

        with open(self.global_config['data_dir'], 'r') as file:
            self.data = [
                json.loads(line.strip())
                for line in file.readlines()
                if json.loads(line.strip()).get('split') == self.global_config['split']
            ]

    def run(self):
        for current_problem in range(self.worker_idx, len(self.data), self.config['num_procs']):
            self.logger.info(
                f"Working on problem {current_problem}")
            # log the config
            self.logger.info(f"Config: {self.config}")
            problem = self.data[current_problem]
            informal_prefix = problem['informal_prefix']
            formal_statement = problem['formal_statement']
            PROBLEM_STATEMENT = informal_prefix + formal_statement
            tactic_state = problem['goal']

            state: MetaLeanGameState = MetaLeanGameState.starting_state(
                worker_id=self.worker_id,
                problem=PROBLEM_STATEMENT,
                tactic_state=tactic_state,
                max_depth=self.config['max_depth']
            )

            rollout: WorkerTask = next(state.pre_query())
            self.enqueue_task(rollout)
            time_to_rollout = -time.time()
            rollout_output = self.spin_deque_task(
                channel=self.worker_id
            )[0]
            time_to_rollout += time.time()
            # self.logger.info(f"Time to rollout: {time_to_rollout}")
            next(state.post_query(rollout_output), None)

            states: List[MetaLeanGameState]

            # print(state.next_moves)

            states, distributions, rewards = self_play(
                self,
                state=state,
                num_iters=self.num_iters,
                max_actions=self.max_actions
            )

            MetaLeanGameState.saves(states, os.path.join(
                self.game_data_path, f"{problem['name']}_states.npy"))

            with open(os.path.join(self.game_data_path, f"{problem['name']}_distributions.npy"), "wb") as file:
                # Don't allow pickle, I want this to be a numpy array for sure.
                np.save(file, distributions, allow_pickle=False)
            with open(os.path.join(self.game_data_path, f"{problem['name']}_outcomes.npy"), "wb") as file:
                # Don't allow pickle, I want this to be a numpy array for sure.
                np.save(file, rewards, allow_pickle=False)

            # save the human printout to a file
            with open(os.path.join(self.output_path, f"{problem['name']}.txt"), 'w', encoding='utf-8') as file:
                for i, state in enumerate(states):
                    file.write(state.human_printout())
            
            upload_to_gcs(os.path.join(self.output_path, f"{problem['name']}.txt"),
                          f"mcts-inference/{self.run_name}/outputs/{problem['name']}.txt")

            self.logger.info(
                f"Finished problem {problem['name']} result: {rewards[-1]}")

    def loop(self):
        pass
