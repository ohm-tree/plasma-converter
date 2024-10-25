"""
This file contains the main entrypoint for linear inference.
It is responsible for running the linear inference algorithm
on a given problem, saving the results to disk, and logging
the results.
"""

import asyncio
import json
import multiprocessing
import os
import pickle
from abc import ABC, abstractmethod
from multiprocessing import sharedctypes

import numpy as np
from wayfinder.uct.self_play import async_self_play

from src.lean.lean_game import LeanGame, LeanState
from src.workers.worker import Worker


class InferenceWorker(Worker, ABC):
    def __init__(self,
                 name: str,
                 worker_type: str,
                 worker_idx: int,
                 global_config: dict,
                 config: dict,
                 run_name: str,
                 queues: dict[str, multiprocessing.Queue],
                 values: dict[str, sharedctypes.Synchronized],
                 **kwargs  # Unused
                 ):
        super().__init__(
            name=name,
            worker_type=worker_type,
            worker_idx=worker_idx,
            queues=queues,
            run_name=run_name,
        )

        self.config = config
        self.global_config = global_config

        self.problem_number: sharedctypes.Synchronized = values['problem_number']

        self.load_problems()

        self.game_data_path = f"data/{run_name}/games/{worker_idx}/"
        os.makedirs(self.game_data_path, exist_ok=True)
        self.output_path = f"outputs/{run_name}/"
        os.makedirs(self.output_path, exist_ok=True)
        self.result_path = f"results/{run_name}/"
        os.makedirs(self.result_path, exist_ok=True)

    def load_problems(self):
        # I live in src/workers/
        WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
        SRC_DIR = os.path.dirname(WORKER_DIR)
        ROOT_DIR = os.path.dirname(SRC_DIR)
        split = self.global_config['split']
        if type(split) == str:
            split = [split]
        self.data = []

        with open(self.global_config['data_dir'], 'r') as file:
            for line in file.readlines():
                problem = json.loads(line.strip())
                if problem.get('split') in split:
                    self.data.append(problem)

    def run(self):
        asyncio.run(self.async_run())

    async def async_run(self):

        # spawn an async task which listens forever
        # for incoming tasks
        listener = asyncio.create_task(
            self.listen(
                channel=self.name
            )
        )

        # for current_problem in range(self.worker_idx, len(self.data), self.config['num_procs']):
        while True:
            with self.problem_number.get_lock():
                current_problem = self.problem_number.value
                if current_problem >= len(self.data):
                    break
                self.problem_number.value += 1
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
                max_depth=40
            )

            results = await self.solve(game)

            if "states" in results:
                LeanState.saves(states=results['states'], filename=os.path.join(
                    self.game_data_path, f"{problem['name']}_states.npy"))
            if "distributions" in results:
                pickle.dump(results['distributions'], open(
                    os.path.join(self.game_data_path, f"{problem['name']}_distributions.npy"), "wb"))

            if "rewards" in results:
                pickle.dump(results['rewards'], open(
                    os.path.join(self.game_data_path, f"{problem['name']}_outcomes.npy"), "wb"))
            if "states" in results:
                # save the human printout to a file
                with open(os.path.join(self.output_path, f"{problem['name']}.txt"), 'w') as file:
                    for i, state in enumerate(results['states']):
                        file.write(state.__str__() + "\n")

            if "result" in results:
                self.logger.info(
                    f"Finished problem {problem['name']} result: {results['result']}")
                with open(os.path.join(self.result_path, f"{problem['name']}.txt"), 'w') as file:
                    file.write(f"Problem: {problem['name']}\n")
                    file.write(f"Split: {problem['split']}\n")
                    file.write(f"Result: {results['result']}\n")

        listener.cancel()

    @abstractmethod
    async def solve(self, game: LeanGame):
        raise NotImplementedError

    def loop(self):
        pass
