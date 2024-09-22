import csv
import json
import logging
import multiprocessing
import os
import random
import sys

import numpy as np
import torch

from src.games.lean_game import LeanGame, LeanGameState
from src.networks.prover_llm import ProverLLM
from src.policies.network_policy import NetworkPolicy
from src.policies.random_policy import RandomPolicy
from src.policies.uct_policy import UCTPolicy
from src.train.self_play import self_play


def setup_logging(task_id, run_name):
    ensure_directory_exists('logs')
    logging.basicConfig(
        filename=f'logs/{run_name}_x{task_id}.log', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_random_game():
    return load_game(random.randint(0, 244))

def load_game(problem_idx):  # TODO: change to problem id?
    # load the problem json
    problem = None
    with open("datasets/minif2f.jsonl", 'r', encoding='utf-8') as file:
        for current_index, line in enumerate(file):
            if current_index == problem_idx:
                problem = json.loads(line)

    print("problem", problem)

    # Load comments
    # TODO: preprocess
    comments = None
    with open("src/sample-data/comments.txt", 'r') as file:
        comments = [line.strip() for line in file.readlines()]

    prompt = problem["informal_prefix"] + problem["formal_statement"]
    goal = problem["goal"]

    print("prompt".center(80, "-"))
    print(prompt)
    print("goal".center(80, "-"))
    print(goal)

    game: LeanGame = LeanGame(
        comment_seeds=comments,
        completion_model=ProverLLM(),
    )
    state = game.start_state(problem=prompt, tactic_state=goal)
    return game, state


def main(
        queue: multiprocessing.Queue,
        completion_queue: multiprocessing.Queue,
        policy_value_queue: multiprocessing.Queue,
        lean_queue: multiprocessing.Queue,
        task_id: int,
        num_tasks: int,
        json_name: str
):
    """
    Entry point for the mcts worker.

    Parameters:
    ----------
    queue: multiprocessing.Queue
        The queue to receive finished tasks from the other processes
    completion_queue: multiprocessing.Queue
        The queue to send requests to the completion process
    policy_value_queue: multiprocessing.Queue
        The queue to send requests to the policy value process
    lean_queue: multiprocessing.Queue
        The queue to send requests to the lean process
    task_id: int
        The id of the worker
    num_tasks: int
        The number of workers
    json_name: str
        The name of the json file to load the configuration from
    """


    with open(f"config/{json_name}.json", "r") as f:
        config = json.load(f)
        MODEL_NAME = config["modelName"]
        MODEL_VARIANT = config["modelVariant"]
        NUM_GROUPS = config["numGroups"]
        NUM_WORKER_TASKS = config["numWorkerTasks"]
        NUM_ITERS = config["numIters"]
        UCT_TRAVERSALS = config["uctTraversals"]

    RUN_NAME = f'{MODEL_NAME}_{MODEL_VARIANT}'
    group_id = task_id // (NUM_WORKER_TASKS // NUM_GROUPS)

    setup_logging(task_id, RUN_NAME)
    logging.info("Starting worker...")

    model_path = f"data/{RUN_NAME}/models"
    game_data_path = f"data/{RUN_NAME}/games/{group_id}/{task_id}"
    ensure_directory_exists(model_path)
    ensure_directory_exists(game_data_path)
    worker_iteration = 0
    # figure out what the worker iteration should be by looking for the output files.
    while True:
        if (os.path.exists(os.path.join(game_data_path, f"{worker_iteration}_states.npy")) and
            os.path.exists(os.path.join(game_data_path, f"{worker_iteration}_distributions.npy")) and
                os.path.exists(os.path.join(game_data_path, f"{worker_iteration}_outcomes.npy"))):
            worker_iteration += 1
        else:
            break

    while True:
        model_iter = 0
        while True:
            model_file = os.path.join(
                model_path, f"{RUN_NAME}_{model_iter}.pt")
            if not os.path.exists(model_file):
                break
            model_iter += 1
        if model_iter == NUM_ITERS:
            logging.info("All models have been created. Exiting...")
            break
        if model_iter == 0:
            logging.info("No models found. Using Random Policy.")
            network_policy = RandomPolicy()
        else:
            logging.info(f"Loading model from {model_iter - 1}.")
            model_file = os.path.join(
                model_path, f"{RUN_NAME}_{model_iter - 1}.pt")
            state_dict = torch.load(model_file)
            model = ProverLLM()
            model.load_state_dict(state_dict)
            network_policy = NetworkPolicy(model)

        policy = UCTPolicy(
            network_policy, num_iters=UCT_TRAVERSALS)

        # load a random game.
        game, start_state = load_random_game()

        print("game", game)
        print("start_state", start_state)

        states, distributions, rewards = self_play(start_state, game, policy)

        LeanGameState.saves(states, os.path.join(
            game_data_path, f"{worker_iteration}_states.npy"))

        with open(os.path.join(game_data_path, f"{worker_iteration}_distributions.npy"), "wb") as file:
            # Don't allow pickle, I want this to be a numpy array for sure.
            np.save(file, distributions, allow_pickle=False)
        with open(os.path.join(game_data_path, f"{worker_iteration}_outcomes.npy"), "wb") as file:
            # Don't allow pickle, I want this to be a numpy array for sure.
            np.save(file, rewards, allow_pickle=False)

        logging.info(f"Iteration {worker_iteration} completed.")
        worker_iteration += 1