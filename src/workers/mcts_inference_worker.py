"""
In this file, we will let a human play a game of Lean.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import queue
import time
from typing import Optional, Union

import numpy as np
from wayfinder.uct.self_play import async_self_play

from src.games.lean_game import LeanGame, LeanState
from src.workers.worker import Worker


class MCTSWorker(Worker):
    def __init__(self,
                 global_config: dict,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: dict[str, multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            name="MCTS" + "_" + str(task_id),
            worker_type="MCTS",
            worker_idx=task_id,
            queues=queues,
            run_name=run_name,
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
        asyncio.run(self.async_run())

    async def async_run(self):
        for current_problem in range(self.worker_idx, len(self.data), self.config['num_procs']):
            self.logger.info(
                f"Working on problem {current_problem}")
            problem = self.data[current_problem]
            informal_prefix = problem['informal_prefix']
            formal_statement = problem['formal_statement']
            PROBLEM_STATEMENT = informal_prefix + formal_statement
            tactic_state = problem['goal']

            game: LeanGame = LeanGame(
                worker=self,
                problem=PROBLEM_STATEMENT,
                tactic_state=tactic_state,
                max_depth=20
            )

            state: LeanState = await game.starting_state()

            states: list[LeanState]

            states, distributions, rewards = await async_self_play(
                self,
                state=state,
                num_iters=self.num_iters,
                max_actions=self.max_actions
            )

            LeanState.saves(states, os.path.join(
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
                    file.write(state.__str__())

            self.logger.info(
                f"Finished problem {problem['name']} result: {rewards[-1]}")

    def loop(self):
        pass
