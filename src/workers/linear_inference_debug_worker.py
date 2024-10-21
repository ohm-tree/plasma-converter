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

from src.games.concurrent import Router
from src.games.lean_game import MetaLeanMove, MetaLeanState
from src.games.lean_game_core import LeanState
from src.workers.types import LinearInferenceDebugWorkerType
from src.workers.worker import TaskType, Worker, WorkerIdentifer, WorkerTask, WorkerType


class LinearInferenceDebugWorker(Worker):
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
                LinearInferenceDebugWorkerType, task_id),
            queues=queues,
            run_name=run_name,
            poison_scream=False
        )

        self.config = config
        self.global_config = global_config

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

        with open(os.path.join(ROOT_DIR, self.global_config['data_dir']), 'r') as file:
            self.data = [
                json.loads(line.strip())
                for line in file.readlines()
                if json.loads(line.strip()).get('split') == self.global_config['split']
            ]

    def run(self):
        for current_problem in range(self.worker_idx, len(self.data), self.config['num_procs']):
            self.logger.info(
                f"Working on problem {current_problem}")
            problem = self.data[current_problem]
            informal_prefix = problem['informal_prefix']
            formal_statement = problem['formal_statement']
            PROBLEM_STATEMENT = informal_prefix + formal_statement
            tactic_state = problem['goal']

            state: MetaLeanState = MetaLeanState.starting_state(
                worker_id=self.worker_id,
                problem=PROBLEM_STATEMENT,
                tactic_state=tactic_state
            )

            context_input: WorkerTask = next(state.pre_comments())
            self.enqueue_task(context_input)
            time_to_context = -time.time()
            context_output = self.spin_deque_task(
                channel=self.worker_id
            )[0]
            time_to_context += time.time()
            self.logger.info(f"Time to context: {time_to_context}")
            next(state.post_comments(context_output), None)

            router = Router(self)

            done = False

            def callback_done():
                nonlocal done
                done = True
                yield from ()

            while not state.terminal():
                done = False

                self.logger.info(state.human_printout())
                action = state.get_active_move(0)
                state = state.next_state(action)

                ### Enter the gungeon ###
                router.startup(state, callback_done)
                while not done:
                    router.tick(blocking=True)
                    self.logger.info(router.debug())

            self.logger.info(state.human_printout())

            # save the human printout to a file
            with open(os.path.join(self.output_path, f"{problem['name']}.txt"), 'w') as file:
                file.write(state.human_printout())

            self.logger.info(
                f"Finished problem {problem['name']} result: {state.reward()}")

    def loop(self):
        pass
