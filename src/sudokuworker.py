import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch

from src.games.sudoku import SudokuGame, SudokuGameState
from src.networks.sudokunet import SudokuCNN
from src.policies.monte_carlo_policy import MonteCarloPolicy
from src.policies.network_policy import NetworkPolicy
from src.policies.policy import Policy
from src.policies.random_policy import RandomPolicy
from src.policies.uct_policy import UCTPolicy
from src.train.self_play import self_play

# get json_name from the third argv
task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])
json_name = sys.argv[3]

with open(f"./config/{json_name}.json", "r") as f:
    config = json.load(f)
    MODEL_NAME = config["modelName"]
    MODEL_VARIANT = config["modelVariant"]
    NUM_GROUPS = config["numGroups"]
    NUM_WORKER_TASKS = config["numWorkerTasks"]
    NUM_ITERS = config["numIters"]

    WORLD_SIZE = config["worldSize"]
    WORKER_TIME_TO_KILL = config["workerTimeToKill"]
    USE_DDP = config["use_ddp"]

    SYNC = config["sync"]
    LINEAR_WEIGHTING = config["linearWeighting"]
    NUM_TRAIN_SAMPLES = config["numTrainSamples"]
    NUM_SAVE_SAMPLES = config["numSaveSamples"]

    BATCH_SIZE = config["batchSize"]
    LR_INIT = config["lrInit"]

    UCT_TRAVERSALS = config["uctTraversals"]


RUN_NAME = f'{MODEL_NAME}_{MODEL_VARIANT}'

group_id = task_id // (NUM_WORKER_TASKS // NUM_GROUPS)


def setup_logging(task_id):
    logging.basicConfig(
        filename=f'logs/{RUN_NAME}_x{task_id}.log', level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_data(file_path, data):
    with open(file_path, "wb") as file:
        np.save(file, data)


def load_random_game(difficulty=1.0):
    return load_game(random.randint(0, int(8e6)), difficulty)


def load_game(game_id, difficulty):
    # Load the CSV file
    batch = game_id // 1000
    sub_id = game_id % 1000
    print(batch, sub_id)
    # print the cwd
    with open(f'data/sudoku_files/sudoku_{batch}.csv', 'r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)
        # Read the rows into a list
        games = [row for row in reader]
    # Get the specific game
    problem, solution = games[sub_id]

    # the final problem should contain (difficulty) * problem + (1 - difficulty) * solution
    final_problem = ""
    for i, j in zip(problem, solution):
        if random.random() < difficulty:
            final_problem += i
        else:
            final_problem += j

    # we only actually care about the problem, not the solution.
    return SudokuGameState.from_string(final_problem)


def main():
    setup_logging(task_id)
    logging.info(f"Starting worker...")

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
            model = SudokuCNN()
            model.load_state_dict(state_dict)
            network_policy = NetworkPolicy(model)

        policy = UCTPolicy(
            network_policy, num_iters=UCT_TRAVERSALS)

        # load a random game.
        game = load_random_game(
            difficulty=1 - np.exp(-worker_iteration / 100))

        print(game)

        states, distributions, rewards = self_play(game, SudokuGame(), policy)

        save_data(os.path.join(game_data_path,
                  f"{worker_iteration}_states.npy"), states)
        save_data(os.path.join(game_data_path,
                  f"{worker_iteration}_distributions.npy"), distributions)
        save_data(os.path.join(game_data_path,
                  f"{worker_iteration}_outcomes.npy"), rewards)

        logging.info(f"Iteration {worker_iteration} completed.")
        worker_iteration += 1


if __name__ == "__main__":
    main()
