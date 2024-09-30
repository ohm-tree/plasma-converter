"""
self_play.py

This module contains the self-play function, which plays a game between two policies
and returns the game states and action distributions, as well as the final result.

It also contains a larger function which generates a dataset of self-play games.
"""

import multiprocessing
import queue
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm

from src.games.lean_game import LeanGame, LeanGameState
from src.policies.policy import Policy
from src.uct.uct_alg import uct_search
from src.uct.uct_node import UCTNode


def self_play(worker_id: int, state: LeanGameState, game: LeanGame, num_iters: int,
              worker_queue: multiprocessing.Queue,
              completion_queue: multiprocessing.Queue,
              context_queue: multiprocessing.Queue,
              lean_queue: multiprocessing.Queue,
              ) -> Tuple[List[LeanGameState], List[np.ndarray], float]:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
    """

    states: List[Any] = []
    distributions: List[np.ndarray] = []

    move_count = 0

    # Edge case: on the very first move, the completions are not available yet.
    # Send those in.

    context_queue.put(
        {
            'mcts_worker_id': worker_id,
            'task_id': 0,
            'task_input': state.pre_comments(),
            'type': 'context'
        }
    )

    context_output = None
    while context_output is None:
        try:
            context_output = worker_queue.get_nowait()
        except queue.Empty:
            context_output = None
            pass

    assert context_output['type'] == 'policy_value'
    assert context_output['task_id'] == 0
    state.post_comments(context_output['task_output']['comments'])

    # first, we need to comment the state right away.

    while not game.is_terminal(state):
        states.append(state)

        """
        TODO: Fast Playouts would be implemented here.
        """

        # TODO: subtree reuse.
        root = UCTNode(game, state, -1, init_type="zero")

        distribution, _ = uct_search(
            worker_id,
            worker_queue=worker_queue,
            completion_queue=completion_queue,
            context_queue=context_queue,
            lean_queue=lean_queue,
            game=game,
            root=root,
            num_iters=num_iters
        )

        # TODO: more configurations possible for uct_search, not used right now.

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
