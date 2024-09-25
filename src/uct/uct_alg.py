"""
uct_alg.py

This module contains functions for running the UCT algorithm.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

import multiprocessing
import time
from typing import Dict, Tuple

import numpy as np

from src.games.lean_game import LeanGame, LeanGameState
from src.policies.policy import Policy
from src.uct.uct_node import UCTNode


def uct_search(
    worker_id: int,
    queue: multiprocessing.Queue,
    completion_queue: multiprocessing.Queue,
    policy_value_queue: multiprocessing.Queue,
    lean_queue: multiprocessing.Queue,
    game: LeanGame,
    game_state: LeanGameState,
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

    # set root action to -1 so we can identify it and add noise
    root = UCTNode(game, game_state, -1, init_type=init_type)

    completion_waiting: Dict[int, LeanGameState] = {}
    policy_value_waiting: Dict[int, LeanGameState] = {}
    lean_waiting: Dict[int, LeanGameState] = {}

    for _ in range(num_iters):
        # greedily select leaf with given exploration parameter
        leaf: UCTNode = root.select_leaf_no_virtual_loss(c)

        if leaf.is_terminal:
            root.select_leaf(c)  # Apply the virtual loss this time.

            # compute the value estimate of the player at the terminal leaf
            value_estimate: float = game.reward(leaf.game_state)
            # Immediately backup the value estimate along the path to the root
            leaf.backup(value_estimate)

        else:
            if hash(leaf) in completion_waiting:
                # This annoys us, because
                # it is an already-visited node.
                # We simply yield control from this process for a
                # bit while we wait for the lean worker to finish.
                time.sleep(10)
            else:
                root.select_leaf(c)  # Apply the virtual loss this time.

                # Add the child priors and value estimate to the completion queue!
                # tasks should take the form
                # {
                #   'worker_id': int, # The worker task id that generated this task.
                #   'completion_task_id': int, # The specific completion task id of this task.
                #   'task': str # The task to complete, a string prompt.
                # }

                completion_queue.put(
                    {
                        'worker_id': worker_id,
                        'task_id': hash(leaf),
                        'task_input': leaf.game_state.pre_LLM_rollout(),
                        'task': 'completion'
                    }
                )
                completion_waiting[hash(leaf)] = leaf.game_state
        # Check for completed leaves.

        # Load any results from the completion queue, lean queue, and policy_value_queue.
        # and enqueue them all to the lean_queue and policy_value_queue.
        while not queue.empty():
            result = queue.get()
            if result['task'] == 'completion':
                # Find the node that requested this completion.
                node: UCTNode = completion_waiting[result['task_id']]
                # Update the node with the completion.
                node.game_state.post_LLM_rollout(result['output'])

                # Enqueue the node to the lean queue.
                lean_queue.put(
                    {
                        'worker_id': worker_id,
                        'task_id': hash(node),
                        'task_input': node.pre_process(),
                        'task': 'lean'
                    }
                )

                # Enqueue the node to the policy_value_queue.
                policy_value_queue.put(
                    {
                        'worker_id': worker_id,
                        'task_id': hash(node),
                        'task_input': node.pre_policy_value(),
                        'task': 'policy_value'
                    }
                )
                completion_waiting.pop(result['task_id'])

            elif result['task'] == 'policy_value':
                # Find the node that requested this policy value.
                node: UCTNode = policy_value_waiting[result['task_id']]
                # Update the node with the policy value.
                node.expand(result['policy'], result['value'], train)
                node.backup(result['value'])

                completion_waiting.pop(result['task_id'])

            elif result['task'] == 'lean':
                # Find the node that requested this lean.
                node: UCTNode = lean_waiting[result['task_id']]

                # Update the node with the lean.
                node.is_processed = True
                node.game_state.post_process(result['output'])

                completion_waiting.pop(result['task_id'])

    print("Number visits", [i.item()
          for i in root.child_number_visits if i > 0])
    print("Prior policy", [i.item() for i in root.child_priors if i > 0])
    print("Q values", [i.item() for i, j in zip(
        root.child_Q(), root.child_number_visits) if j > 0])
    print("U values", [i.item() for i, j in zip(
        root.child_U(), root.child_number_visits) if j > 0])

    return root.child_number_visits / np.sum(root.child_number_visits), root.child_Q()[root.child_number_visits.argmax()]
