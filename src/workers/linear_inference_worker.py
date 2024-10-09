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
from src.games.lean_game_core import LeanGameState
from src.train.self_play import self_play
from src.workers.types import (
    CompletionTaskType,
    LeanTaskType,
    LinearInferenceWorkerType,
    PolicyValueTaskType,
)
from src.workers.worker import TaskType, Worker, WorkerIdentifer, WorkerType


class LinearInferenceWorker(Worker):
    def __init__(self,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: Dict[Union[TaskType, WorkerIdentifer], multiprocessing.Queue],
                 ):
        super().__init__(
            worker_id=WorkerIdentifer(
                LinearInferenceWorkerType, task_id),
            queues=queues,
            run_name=run_name,
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

            state: MetaLeanGameState = MetaLeanGameState.starting_state(
                worker_id=self.worker_id,
                problem=PROBLEM_STATEMENT,
                tactic_state=tactic_state
            )

            while not state.terminal():
                self.logger.info(state.human_printout())
                action = 0
                state = state.next_state(action)
                input_data = next(state.pre_LLM_rollout())
                self.enqueue_task(input_data)
                time_to_completion = -time.time()
                completion_output = next(self.spin_deque_task(
                    task_type=CompletionTaskType
                ))
                time_to_completion += time.time()
                self.logger.info(f"Time to completion: {time_to_completion}")
                lean4_input = next(state.post_LLM_rollout(completion_output))
                self.enqueue_task(lean4_input)
                time_to_lean = -time.time()
                lean_output = next(self.spin_deque_task(
                    task_type=LeanTaskType
                ))
                time_to_lean += time.time()
                self.logger.info(f"Time to lean: {time_to_lean}")
                context_input = next(state.post_process(lean_output))
                self.enqueue_task(context_input)
                time_to_context = -time.time()
                context_output = next(self.spin_deque_task(
                    task_type=PolicyValueTaskType
                ))
                time_to_context += time.time()
                self.logger.info(f"Time to context: {time_to_context}")
                state.post_comments(context_output)

            # save the human printout to a file
            os.makedirs("outputs/distributed_run/", exist_ok=True)
            with open(f"outputs/distributed_run/{problem['name']}.txt", 'w') as file:
                file.write(state.human_printout())

            self.logger.info(
                f"Finished problem {problem['name']} result: {state.reward()}")
