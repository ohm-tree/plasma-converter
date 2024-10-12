"""
self_play.py

This module contains the self-play function, which plays a game between two policies
and returns the game states and action distributions, as well as the final result.

It also contains a larger function which generates a dataset of self-play games.
"""

import logging
import multiprocessing
import queue
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm

from src.games.game import ConcurrentGameState, ConcurrentMetaGameState
from src.uct.uct_alg import uct_search
from src.uct.uct_node import UCTNode
from src.workers import *


def self_play(
    self: Worker,
    state: ConcurrentMetaGameState,
    num_iters: int,
    max_actions: int = 10
) -> Tuple[List[ConcurrentMetaGameState], List[np.ndarray], float]:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
    """

    states: List[Any] = []
    distributions: List[np.ndarray] = []

    move_count = 0

    # Send those in.
    root = UCTNode(self.worker_id, state, -1,
                   init_type="zero", max_actions=max_actions)

    next(root.backprop_and_expand(), None)

    # root.is_processed = True
    states.append(root.game_state)
    while not root.game_state.terminal():
        self.logger.info("Move: " + str(move_count))
        move_count += 1
        self.logger.info(root.game_state.human_printout())
        """
        TODO: Fast Playouts would be implemented here.
        """
        winning_node: UCTNode
        distribution, _, winning_node = uct_search(
            self,
            root=root,
            num_iters=num_iters
        )

        if winning_node is not None:
            root = winning_node
            break
        distributions.append(distribution)
        self.logger.info(f"Action distribution: {distribution}")

        action = np.random.choice(len(distribution), p=distribution)
        root = root.children[action]
        # set root parent to None so that it knows it is the root.
        root.root()
        states.append(root.game_state)

    self.logger.info("Move: " + str(move_count))
    self.logger.info(root.game_state.human_printout())

    # The reward for all states in the tree is the reward of the final state.

    final_reward = root.game_state.reward()
    if winning_node is not None:
        self.logger.info(
            "Game finished early with reward: " + str(final_reward))
    else:
        self.logger.info(
            f"Game finished after {move_count} moves with reward: {final_reward}")
    rewards = [final_reward for _ in states]
    return states, distributions, rewards
