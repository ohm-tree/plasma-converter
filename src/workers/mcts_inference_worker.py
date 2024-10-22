"""
This file contains the main entrypoint for the MCTS algorithm.
It is responsible for running the MCTS algorithm on a given problem,
saving the results to disk, and logging the results.
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
from src.workers.inference_worker import InferenceWorker


class MCTSWorker(InferenceWorker):
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
            global_config=global_config,
            config=config,
            queues=queues,
            run_name=run_name,
        )

    async def solve(self, game: LeanGame) -> None:
        state: LeanState = await game.starting_state()

        states: list[LeanState]

        states, distributions, rewards = await async_self_play(
            self,
            state=state,
            num_iters=self.num_iters,
            max_actions=self.max_actions
        )

        return {
            'states': states,
            'distributions': distributions,
            'rewards': rewards,
            "result": rewards[-1]
        }

    def loop(self):
        pass
