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
from src.uct.uct_alg import uct_search
from src.uct.uct_node import UCTNode
from src.workers.mcts_inference_worker import (
    CompletionTaskType,
    LeanTaskType,
    MCTSWorker,
    MCTSWorkerType,
    PolicyValueTaskType,
)


def self_play(
    self: MCTSWorker,
    state: LeanGameState,
    game: LeanGame,
    num_iters: int,
) -> Tuple[List[LeanGameState], List[np.ndarray], float]:
    """
    Play a game using a policy, and return the game states, action distributions, and final reward.
    """

    states: List[Any] = []
    distributions: List[np.ndarray] = []

    move_count = 0

    # Edge case: on the very first move, the completions are not available yet.
    # Send those in.

    self.enqueue_task(
        obj=state.pre_comments(),
        task_idx=0,
        task_type=PolicyValueTaskType
    )

    context_output = self.spin_deque_task(PolicyValueTaskType)[0]
    assert context_output.task_id.task_idx == 0

    state.post_comments(context_output.response['comments'])

    root = UCTNode(game, state, -1, init_type="zero")

    # first, we need to comment the state right away.
    self.logger.info(
        f"Received policy: {context_output['task_output']['policy']}")
    self.logger.info(
        f"Received value: {context_output['task_output']['value']}")

    root.expand(context_output['task_output']['policy'],
                context_output['task_output']['value'], train=True)
    root.backup(context_output['task_output']['value'])

    root.is_processed = True
    states.append(root.game_state)
    while not game.is_terminal(root.game_state):

        self.logger.info("Move: " + str(move_count))
        self.logger.info(root.game_state.human_printout())
        """
        TODO: Fast Playouts would be implemented here.
        """
        winning_node: UCTNode
        distribution, _, winning_node = uct_search(
            self,
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
        self.logger.info(f"Action distribution: {distribution}")

        action = np.random.choice(len(distribution), p=distribution)
        root = root.children[action]
        # set root parent to None so that it knows it is the root.
        root.root()
        states.append(root.game_state)
        move_count += 1

    # The reward for all states in the tree is the reward of the final state.
    if winning_node is not None:
        final_reward = game.reward(winning_node.game_state)
        self.logger.info(
            "Game finished early with reward: " + str(final_reward))
    else:
        final_reward = game.reward(root.game_state)
        self.logger.info(
            f"Game finished after {move_count} moves with reward: {final_reward}")
    rewards = [final_reward for _ in states]
    return states, distributions, rewards
