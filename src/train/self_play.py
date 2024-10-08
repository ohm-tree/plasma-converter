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

from src.games.lean_game import LeanGame, LeanGameState
from src.policies.policy import Policy
from src.uct.uct_alg import uct_search
from src.uct.uct_node import UCTNode


def self_play(worker_id: int, state: LeanGameState, game: LeanGame, num_iters: int,
              logger: logging.Logger,
              worker_queue: multiprocessing.Queue,
              global_completion_queue: multiprocessing.Queue,
              global_context_queue: multiprocessing.Queue,
              global_lean_queue: multiprocessing.Queue,
              ) -> Tuple[List[LeanGameState], List[np.ndarray], float]:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
    """

    states: List[Any] = []
    distributions: List[np.ndarray] = []

    move_count = 0

    # Edge case: on the very first move, the completions are not available yet.
    # Send those in.

    global_context_queue.put(
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

    # TODO: subtree reuse.
    root = UCTNode(game, state, -1, init_type="zero")

    # first, we need to comment the state right away.
    logger.info(
        f"Received policy: {context_output['task_output']['policy']}")
    logger.info(
        f"Received value: {context_output['task_output']['value']}")

    root.expand(context_output['task_output']['policy'],
                context_output['task_output']['value'], train=True)
    root.backup(context_output['task_output']['value'])

    root.is_processed = True
    states.append(root.game_state)
    while not game.is_terminal(root.game_state):

        logger.info("Move: " + str(move_count))
        logger.info(root.game_state.human_printout())
        """
        TODO: Fast Playouts would be implemented here.
        """

        distribution, _, winning_node = uct_search(
            logger,
            worker_id,
            worker_queue=worker_queue,
            global_completion_queue=global_completion_queue,
            global_context_queue=global_context_queue,
            global_lean_queue=global_lean_queue,
            game=game,
            root=root,
            num_iters=num_iters
        )

        # TODO: more configurations possible for uct_search, not used right now.

        """
        TODO: In MCTS algorithms, people sometimes change up the temperature right here,
        to sharpen the training distribution. This is something we could try.
        """

        if winning_node is not None:
            root = winning_node
            break
        distributions.append(distribution)
        logger.info(f"Action distribution: {distribution}")

        action = np.random.choice(len(distribution), p=distribution)
        root = root.children[action]
        # set root parent to None so that it knows it is the root.
        root.root()
        states.append(root.game_state)
        move_count += 1

    # The reward for all states in the tree is the reward of the final state.
    if winning_node is not None:
        final_reward = game.reward(winning_node.game_state)
        logger.info("Game finished early with reward: " + str(final_reward))
    else:
        final_reward = game.reward(root.game_state)
        logger.info(
            f"Game finished after {move_count} moves with reward: {final_reward}")
    rewards = [final_reward for _ in states]
    return states, distributions, rewards
