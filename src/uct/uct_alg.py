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


def uct_search(
    logger: logging.Logger,
    worker_id: int,
    worker_queue: multiprocessing.Queue,
    global_completion_queue: multiprocessing.Queue,
    global_context_queue: multiprocessing.Queue,
    global_lean_queue: multiprocessing.Queue,
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

    victorious_death = False
    winning_node = None
    # set root action to -1 so we can identify it and add noise

    completion_waiting: Dict[int, Tuple[LeanGameState, float]] = {}
    context_waiting: Dict[int, Tuple[LeanGameState, float]] = {}
    lean_waiting: Dict[int, Tuple[LeanGameState, float]] = {}
    iters = 0

    sum_completion_time = 0
    sum_context_time = 0
    sum_lean_time = 0
    total_completion = 0
    total_context = 0
    total_lean = 0

    absolute_start_time = time.time()
    dead_time = 0

    while not victorious_death:
        dead_time_start = time.time()
        activity = False
        if iters >= num_iters and len(completion_waiting) == 0 and len(context_waiting) == 0 and len(lean_waiting) == 0:
            break
        if iters < num_iters and len(completion_waiting) + len(context_waiting) + len(lean_waiting) < 10:
            # greedily select leaf with given exploration parameter
            leaf: UCTNode = root.select_leaf_no_virtual_loss(c)

            assert (not leaf.is_expanded) or (leaf.is_terminal)

            # Problem: we don't know if a leaf is terminal until we lean4-verify it!
            if leaf.game_state.step >= LeanGameStateStep.PROCESSED and leaf.is_terminal:
                activity = True
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
                activity = True

                # Add the child priors and value estimate to the completion queue!
                # tasks should take the form
                # {
                #   'worker_id': int, # The worker task id that generated this task.
                #   'completion_task_id': int, # The specific completion task id of this task.
                #   'task': str # The task to complete, a string prompt.
                # }

                state: LeanGameState = leaf.game_state

                global_completion_queue.put(
                    {
                        'mcts_worker_id': worker_id,
                        'completion_task_id': hash(leaf),
                        'task': state.pre_LLM_rollout(),
                        'type': 'completion'
                    }
                )
                completion_waiting[hash(leaf)] = (leaf, time.time())
                iters += 1
        # Check for completed leaves.

        # Load any results from the completion queue, lean queue, and context_queue.
        # and enqueue them all to the lean_queue and context_queue.
        if worker_queue.empty() and (iters >= num_iters or len(completion_waiting) + len(context_waiting) + len(lean_waiting) >= 10):
            # just be patient i guess. don't clog up cpu spinning.
            time.sleep(1)
        while not worker_queue.empty():
            activity = True
            result = worker_queue.get()
            node: UCTNode
            time_init: float
            if result['type'] == 'completion':
                # Find the node that requested this completion.
                node, time_init = completion_waiting.pop(
                    result['completion_task_id'])

                time_taken = time.time() - time_init

                logger.info("Received completion output, took " +
                            str(time_taken) + " seconds.")
                sum_completion_time += time_taken
                total_completion += 1

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
                lean_waiting[hash(node)] = (node, time.time())

            elif result['type'] == 'lean':
                # Find the node that requested this lean.
                node, time_init = lean_waiting.pop(result['lean_task_id'])

                time_taken = time.time() - time_init

                logger.info("Received lean output, took " +
                            str(time_taken) + " seconds.")
                sum_lean_time += time_taken
                total_lean += 1

                # Update the node with the lean.
                state: LeanGameState = node.game_state
                state.post_process(result['result'])
                node.is_processed = True

                if node.is_terminal:
                    # compute the value estimate of the player at the terminal leaf
                    value_estimate: float = game.reward(state)
                    # Immediately backup the value estimate along the path to the root
                    node.backup(value_estimate)

                    if value_estimate == 1.0:
                        logger.info(
                            "We think we just won! Here was the lean output:")
                        logger.info(result['result'])
                        victorious_death = True
                        winning_node = node

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
                    context_waiting[hash(node)] = (node, time.time())

            elif result['type'] == 'policy_value':
                # Find the node that requested this policy value.
                node, time_init = context_waiting.pop(result['task_id'])

                time_taken = time.time() - time_init

                logger.info("Received context output, took " +
                            str(time_taken) + " seconds.")
                sum_context_time += time_taken
                total_context += 1

                # Update the node with the policy value.
                state: LeanGameState = node.game_state
                state.post_comments(result['task_output']['comments'])

                # first, we need to comment the state right away.
                logger.info(
                    f"Received policy: {result['task_output']['policy']}")
                logger.info(
                    f"Received value: {result['task_output']['value']}")
                policy = np.array(result['task_output']['policy'])
                node.expand(policy,
                            result['task_output']['value'], train)
                node.backup(result['task_output']['value'])
        if activity:
            logger.info(f"Number of iterations: {iters}")
            logger.info(
                f"Number of completion_waiting: {len(completion_waiting)}")
            logger.info(f"Number of context_waiting: {len(context_waiting)}")
            logger.info(f"Number of lean_waiting: {len(lean_waiting)}")

            logger.info(f"Number visits: {root.child_number_visits}")
            logger.info(f"Prior policy: {root.child_priors}")
            logger.info(f"Q values: {root.child_Q()}")
            logger.info(f"U values: {root.child_U()}")
            if total_completion > 0:
                logger.info(
                    f"Total completion time: {sum_completion_time}, average: {sum_completion_time / total_completion}")
            if total_context > 0:
                logger.info(
                    f"Total context time: {sum_context_time}, average: {sum_context_time / total_context}")
            if total_lean > 0:
                logger.info(
                    f"Total lean time: {sum_lean_time}, average: {sum_lean_time / total_lean}")

            logger.info(
                f"Dead time: {dead_time}, time elapsed: {time.time() - absolute_start_time}")

        if not activity:
            dead_time += time.time() - dead_time_start
    return root.child_number_visits / np.sum(root.child_number_visits), root.child_Q()[root.child_number_visits.argmax()], winning_node
