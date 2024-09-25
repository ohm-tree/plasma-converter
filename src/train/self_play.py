"""
self_play.py

This module contains the self-play function, which plays a game between two policies
and returns the game states and action distributions, as well as the final result.

It also contains a larger function which generates a dataset of self-play games.
"""

import multiprocessing
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm

from src.games.game import Game, GameState
from src.policies.policy import Policy


def self_play(state: GameState, game: Game, policy: Policy,
              queue: multiprocessing.Queue,
              completion_queue: multiprocessing.Queue,
              policy_value_queue: multiprocessing.Queue,
              lean_queue: multiprocessing.Queue,
              ) -> Tuple[List[GameState], List[np.ndarray], float]:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
    """

    states: List[Any] = []
    distributions: List[np.ndarray] = []

    move_count = 0

    while not game.is_terminal(state):
        states.append(state)

        """
        TODO: Fast Playouts would be implemented here.
        """

        distribution, _ = policy.action(game, state)

        """
        TODO: In MCTS algorithms, people sometimes change up the temperature right here,
        to sharpen the training distribution. This is something we could try.
        """

        distributions.append(distribution)

        action = np.random.choice(len(distribution), p=distribution)
        state = game.next_state(state, action)

        move_count += 1

    # The reward for all states in the tree is the reward of the final state.
    final_reward = game.reward(state)
    print(
        f"Game finished after {move_count} moves with reward: {final_reward}")
    rewards = [final_reward for _ in states]
    return states, distributions, rewards
