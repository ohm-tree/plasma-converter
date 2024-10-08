"""
uct_alg.py

This module contains functions for running the UCT algorithm.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

import logging
import multiprocessing
import time
from typing import Dict, Tuple

import numpy as np

from src.games.lean_game import LeanGame, LeanGameState, LeanGameStateStep
from src.uct.uct_node import UCTNode
from src.workers.mcts_inference_worker import (
    CompletionTaskType,
    LeanTaskType,
    MCTSWorker,
    MCTSWorkerType,
    PolicyValueTaskType,
)


def uct_search(
    self: MCTSWorker,
    game: LeanGame,
    root: UCTNode,
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

    completion_waiting: Dict[int, LeanGameState] = {}
    context_waiting: Dict[int, LeanGameState] = {}
    lean_waiting: Dict[int, LeanGameState] = {}
    iters = 0

    while True:
        self.logger.info(f"Number of iterations: {iters}")
        self.logger.info(
            f"Number of completion_waiting: {len(completion_waiting)}")
        self.logger.info(f"Number of context_waiting: {len(context_waiting)}")
        self.logger.info(f"Number of lean_waiting: {len(lean_waiting)}")

        self.logger.info(f"Number visits: {root.child_number_visits}")
        self.logger.info(f"Prior policy: {root.child_priors}")
        self.logger.info(f"Q values: {root.child_Q()}")
        self.logger.info(f"U values: {root.child_U()}")

        if iters >= num_iters and len(completion_waiting) == 0 and len(context_waiting) == 0 and len(lean_waiting) == 0:
            break
        if iters < num_iters:
            # greedily select leaf with given exploration parameter
            leaf: UCTNode = root.select_leaf_no_virtual_loss(c)

            assert (not leaf.is_expanded) or (leaf.is_terminal)

            # Problem: we don't know if a leaf is terminal until we lean4-verify it!
            if leaf.game_state.step >= LeanGameStateStep.PROCESSED and leaf.is_terminal:
                root.select_leaf(c)  # Apply the virtual loss this time.

                # compute the value estimate of the player at the terminal leaf
                value_estimate: float = game.reward(leaf.game_state)
                # Immediately backup the value estimate along the path to the root
                leaf.backup(value_estimate)
                iters += 1

            elif hash(leaf) in completion_waiting or hash(leaf) in context_waiting or hash(leaf) in lean_waiting:
                assert LeanGameStateStep.INITIALIZED <= leaf.game_state.step <= LeanGameStateStep.PROCESSED
                # This annoys us, because
                # it is an already-visited node.
                # We simply yield control from this process for a
                # bit while we wait for the lean worker to finish.
                time.sleep(1)
            else:
                # We have absolutely never seen this leaf before.
                assert leaf.game_state.step == LeanGameStateStep.INITIALIZED
                root.select_leaf(c)  # Apply the virtual loss this time.

                # Add the child priors and value estimate to the completion queue!
                # tasks should take the form
                # {
                #   'worker_id': int, # The worker task id that generated this task.
                #   'completion_task_id': int, # The specific completion task id of this task.
                #   'task': str # The task to complete, a string prompt.
                # }

                state: LeanGameState = leaf.game_state

                self.enqueue_task(
                    obj=state.pre_LLM_rollout(),
                    task_idx=hash(leaf),
                    task_type=CompletionTaskType
                )
                completion_waiting[hash(leaf)] = leaf
                iters += 1
        # Check for completed leaves.

        # Load any results from the completion queue, lean queue, and context_queue.
        # and enqueue them all to the lean_queue and context_queue.

        while not worker_queue.empty():
            result = worker_queue.get()
            if result['type'] == 'completion':
                # Find the node that requested this completion.
                node: UCTNode = completion_waiting.pop(
                    result['completion_task_id'])
                # Update the node with the completion.
                state: LeanGameState = node.game_state
                state.post_LLM_rollout(result['output'])

                # Enqueue the node to the lean queue.
                global_lean_queue.put(
                    {
                        'mcts_worker_id': worker_id,
                        'lean_task_id': hash(node),
                        'task': state.pre_process(),
                        'type': 'lean'
                    }
                )
                lean_waiting[hash(node)] = node

            elif result['type'] == 'lean':
                # Find the node that requested this lean.
                node: UCTNode = lean_waiting.pop(result['lean_task_id'])

                # Update the node with the lean.
                state: LeanGameState = node.game_state
                state.post_process(result['result'])
                node.is_processed = True

                if node.is_terminal:
                    # compute the value estimate of the player at the terminal leaf
                    value_estimate: float = game.reward(state)
                    # Immediately backup the value estimate along the path to the root
                    node.backup(value_estimate)
                else:
                    # Enqueue the node to the context_queue.
                    global_context_queue.put(
                        {
                            'mcts_worker_id': worker_id,
                            'task_id': hash(node),
                            'task_input': state.pre_comments(),
                            'type': 'context'
                        }
                    )
                    context_waiting[hash(node)] = node

            elif result['type'] == 'policy_value':
                # Find the node that requested this policy value.
                node: UCTNode = context_waiting.pop(result['task_id'])
                # Update the node with the policy value.
                state: LeanGameState = node.game_state
                state.post_comments(result['task_output']['comments'])

                # first, we need to comment the state right away.
                self.logger.info(
                    f"Received policy: {result['task_output']['policy']}")
                self.logger.info(
                    f"Received value: {result['task_output']['value']}")
                policy = np.array(result['task_output']['policy'])
                node.expand(policy,
                            result['task_output']['value'], train)
                node.backup(result['task_output']['value'])

    # print("Number visits", [i.item()
    #       for i in root.child_number_visits if i > 0])
    # print("Prior policy", [i.item() for i in root.child_priors if i > 0])
    # print("Q values", [i.item() for i, j in zip(
    #     root.child_Q(), root.child_number_visits) if j > 0])
    # print("U values", [i.item() for i, j in zip(
    #     root.child_U(), root.child_number_visits) if j > 0])
    print("Number visits", root.child_number_visits)
    print("Prior policy", root.child_priors)
    print("Q values", root.child_Q())
    print("U values", root.child_U())
    return root.child_number_visits / np.sum(root.child_number_visits), root.child_Q()[root.child_number_visits.argmax()]
