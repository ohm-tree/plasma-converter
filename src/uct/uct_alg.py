"""
uct_alg.py

This module contains functions for running the UCT algorithm.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterator, List, Tuple, TypeVar

import numpy as np

from src.games.concurrent import ConcurrentClass, Router
from src.games.game import (
    ConcurrentGameStateType,
    ConcurrentMetaGameState,
    ConcurrentMetaGameStateType,
    GameMoveType,
    MetaGameMoveType,
)
from src.uct.uct_node import UCTNode
from src.workers.types import CompletionTaskType, PolicyValueTaskType
from src.workers.worker import TaskIdentifier, Worker, WorkerResponse, WorkerTask


def uct_search(
    self: "Worker",
    root: UCTNode[ConcurrentMetaGameStateType],
    num_iters: int,
    c: float = 1.0,
    train: bool = True,
    init_type: str = "zero"
) -> Tuple[np.ndarray, float]:
    """
    Perform num_iters iterations of the UCT algorithm from the given game state
    using the exploration parameter c. Return the distribution of visits to each direct child.

    Requires that game_state is a non-terminal state.
    """

    victorious_death = False
    winning_node = None
    iters = 0
    dead_time = 0

    router = Router(self)

    while not victorious_death:
        activity = False
        if iters < num_iters and router.total_active < 10:
            # greedily select leaf with given exploration parameter
            leaf = root.select_leaf_no_virtual_loss(c)

            assert (not leaf.is_expanded) or (leaf.is_terminal)

            # Problem: we don't know if a leaf is terminal until we lean4-verify it!
            if leaf.game_state.ready():
                activity = True
                root.select_leaf(c)  # Apply the virtual loss this time.

                # compute the value estimate of the player at the terminal leaf
                value_estimate: float = leaf.game_state
                # Immediately backup the value estimate along the path to the root
                leaf.backup(value_estimate)
                iters += 1

            elif leaf.game_state.started():
                assert router.contains(hash(leaf))
                # This annoys us, because
                # it is an already-visited node.
                # We simply yield control from this process for a
                # bit while we wait for the lean worker to finish.
                time.sleep(1)
            else:
                # We have absolutely never seen this leaf before.
                root.select_leaf(c)  # Apply the virtual loss this time.
                activity = True

                # Add the child priors and value estimate to the completion queue!
                router.startup(leaf)
                iters += 1
        # Check for completed leaves.

        if activity:
            self.logger.info(f"Number of iterations: {iters}")
            self.logger.info(
                f"Number of completion_waiting: {len(completion_waiting)}")
            self.logger.info(
                f"Number of context_waiting: {len(context_waiting)}")
            self.logger.info(f"Number of lean_waiting: {len(lean_waiting)}")

            self.logger.info(f"Number visits: {root.child_number_visits}")
            self.logger.info(f"Prior policy: {root.child_priors}")
            self.logger.info(f"Q values: {root.child_Q()}")
            self.logger.info(f"U values: {root.child_U()}")
            if total_completion > 0:
                self.logger.info(
                    f"Total completion time: {sum_completion_time}, average: {sum_completion_time / total_completion}")
            if total_context > 0:
                self.logger.info(
                    f"Total context time: {sum_context_time}, average: {sum_context_time / total_context}")
            if total_lean > 0:
                self.logger.info(
                    f"Total lean time: {sum_lean_time}, average: {sum_lean_time / total_lean}")

            self.logger.info(
                f"Dead time: {dead_time}, time elapsed: {time.time() - absolute_start_time}")

    return root.child_number_visits / np.sum(root.child_number_visits), root.child_Q()[root.child_number_visits.argmax()], winning_node
