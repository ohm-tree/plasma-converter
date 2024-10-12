"""
uct_alg.py

This module contains functions for running the UCT algorithm.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

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

    min_log_delta = 10  # 10 seconds between logging to not spam the logs

    absolute_start_time = time.time()
    last_log_time = time.time()
    live_time = 0

    router = Router(self)

    while not victorious_death and iters < num_iters:
        live_time -= time.time()
        while iters < num_iters and router.total_active < 10:
            # greedily select leaf with given exploration parameter
            leaf = root.select_leaf_no_virtual_loss(c)

            assert (not leaf.is_expanded) or (leaf.is_terminal)

            # Problem: we don't know if a leaf is terminal until we lean4-verify it!
            if leaf.game_state.ready():
                self.logger.info(f"Leaf is ready: {leaf.game_state}")
                assert leaf.game_state.terminal()
                root.select_leaf(c)  # Apply the virtual loss this time.

                # compute the value estimate of the player at the terminal leaf
                # value_estimate: float = leaf.game_state
                # Immediately backup the value estimate along the path to the root
                leaf.backup(leaf.game_state.reward())
                iters += 1

                if leaf.game_state.reward() == 1.0:
                    victorious_death = True
                    winning_node = leaf

            elif leaf.started():
                self.logger.info(f"Leaf is started: {leaf.game_state}")
                time.sleep(1)
                # assert router.contains(hash(leaf))
                break
            else:
                self.logger.info(f"Leaf is not ready: {leaf.game_state}")
                # We have absolutely never seen this leaf before.
                root.select_leaf(c)  # Apply the virtual loss this time.

                # Add the child priors and value estimate to the completion queue!
                router.startup(leaf)
                iters += 1
        live_time += time.time()
        # Check for completed leaves.
        router.tick()
        if time.time() - last_log_time > min_log_delta:
            last_log_time = time.time()
            self.logger.info(router.debug())

            self.logger.info(f"Number of iterations: {iters}")

            self.logger.info(f"Number visits: {root.child_number_visits}")
            self.logger.info(f"Prior policy: {root.child_priors}")
            self.logger.info(f"Q values: {root.child_Q()}")
            self.logger.info(f"U values: {root.child_U()}")
            self.logger.info(
                f"Live time: {live_time}, time elapsed: {time.time() - absolute_start_time}")

    # wait for the rest of the leaves to finish
    while router.total_active > 0:
        router.tick()
        time.sleep(1)

    return (
        root.child_number_visits / np.sum(root.child_number_visits),
        root.child_Q()[root.child_number_visits.argmax()],
        winning_node
    )
