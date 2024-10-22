"""
go_controller.py
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from socket import gethostname
from typing import Set, Tuple, Union, list

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

import data_muncher
from src.networks.prover_llm import ProverLLM

"""
TODO: all this passing of references is pretty unwieldly. Either make everything global vars,
or make the whole thing a class.
"""


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# get json_name from the first argv
json_name = sys.argv[1]

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


RUN_NAME = f'{MODEL_NAME}_{MODEL_VARIANT}'

# rank = int(os.environ["SLURM_PROCID"])
if USE_DDP:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are"
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
else:
    rank = 0
    local_rank = 0
    world_size = 1
    print(f"Not using DDP.", flush=True)

logger = logging.getLogger()
os.makedirs(f'logs', exist_ok=True)
logging.basicConfig(
    filename=f'logs/{RUN_NAME}_{rank}.log', level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger.info(
    f'RUN_NAME = {RUN_NAME} RANK = {rank} alive at {time.time()}')

if not USE_DDP:
    non_ddp_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")


def setup_DDP():
    logger.info(f'MASTER_ADDR = {os.environ["MASTER_ADDR"]}')
    logger.info(f'MASTER_PORT = {os.environ["MASTER_PORT"]}')
    logger.info(f'WORLD_SIZE = {os.environ["WORLD_SIZE"]}')
    logger.info(f'RANK = {os.environ["RANK"]}')
    logger.info(f'LOCAL_RANK = {os.environ["LOCAL_RANK"]}')

    logger.info(f'cuda devices: {torch.cuda.device_count()}')

    # initialize the process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    assert int(os.environ["WORLD_SIZE"]) == WORLD_SIZE
    # logger.info(
    #     "Local Rank", os.environ["LOCAL_RANK"], "World Size", WORLD_SIZE)
    logger.info(
        f'Local Rank {os.environ["LOCAL_RANK"]} World Size {WORLD_SIZE}')


def cleanup():
    dist.destroy_process_group()


epochify_time = 0
save_time = 0
trace_time = 0


data_muncher_kwargs = {
    "num_worker_tasks": NUM_WORKER_TASKS,
    "num_groups": NUM_GROUPS,
    "run_name": RUN_NAME,
    "use_linear_wgt": LINEAR_WEIGHTING,
    "worker_ttk": WORKER_TIME_TO_KILL,
    "sync": SYNC,
    "num_train_samples": NUM_TRAIN_SAMPLES,
    "num_save_samples": NUM_SAVE_SAMPLES
}


def train_network(
        network: Union[DDP, ProverLLM], optimizer: optim.Optimizer,
        iteration: int,
        state_tensor: torch.Tensor, distribution_tensor: torch.Tensor,
        outcome_tensor: torch.Tensor, timestamp_tensor: torch.Tensor):
    global epochify_time, save_time, trace_time

    if USE_DDP:
        num_samples = torch.tensor(state_tensor.shape[0], device=local_rank)
        logger.info(
            f"Rank {local_rank} has {num_samples.cpu().item()} samples.")

        # All-reduce the number of batches across all processes; only use the smallest number of batches
        torch.distributed.barrier()
        torch.distributed.all_reduce(num_samples, op=dist.ReduceOp.MIN)
        torch.distributed.barrier()

        num_samples = min(NUM_TRAIN_SAMPLES, num_samples.cpu().item())
        logger.info(
            f"Rank {local_rank} has {num_samples} samples after all-reduce.")

    else:
        num_samples = min(NUM_TRAIN_SAMPLES, state_tensor.shape[0])
        logger.info(f"Rank {local_rank} has {num_samples} samples.")

    state_tensor = state_tensor[-num_samples:]
    distribution_tensor = distribution_tensor[-num_samples:]
    outcome_tensor = outcome_tensor[-num_samples:]
    timestamp_tensor = timestamp_tensor[-num_samples:]

    dataset = TensorDataset(
        state_tensor, distribution_tensor, outcome_tensor, timestamp_tensor)

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(100):
        epochify_time -= time.time()
        train_average_policy_loss, train_average_value_loss = epochify(iteration, 0, epoch,
                                                                       network, dataloader, optimizer, train=True)
        epochify_time += time.time()
        if USE_DDP:
            dist.barrier()
        if epoch % 100 == 0:
            logger.info(
                f"{iteration}.{epoch} Rank {local_rank} " +
                f"Tr Pol: {train_average_policy_loss:.4f}, " +
                f"Tr Val: {train_average_value_loss:.4f}, "
            )

    trace_time -= time.time()

    if rank == 0:
        if USE_DDP:
            state_dict = network.module.policy_value_state_dict()
        else:
            state_dict = network.policy_value_state_dict()
        filepath = f"./data/{RUN_NAME}/models/" + \
            f"{RUN_NAME}_{iteration}.pt"
        torch.save(state_dict, filepath)

    trace_time += time.time()

    logger.info(
        f"Epochify: {epochify_time:.2f}s, Save: {save_time:.2f}s, Trace: {trace_time:.2f}s")


def epochify(iteration: int, group: int, epoch: int,
             network: ProverLLM, train_dataloader: DataLoader,
             optimizer: optim.Optimizer = None, train: bool = True, EPS: float = 1e-8) -> Tuple[float, float]:
    # logger.info(f"Rank {local_rank}:{world_size} entering epochify train={train}")
    if train:
        network.train()
    else:
        network.eval()

    total_policy_loss = 0.0  # These are turned into Tensors on epoch 0.
    total_value_loss = 0.0
    num_batches = 0

    for batch_state, batch_policy, batch_value, batch_timestamp in train_dataloader:
        policy_pred, value_pred = network.policy_and_value(batch_state)

        # Softmax the policy prediction (network returns logits)
        policy_pred = torch.softmax(policy_pred, dim=1)

        # Weighted NLL loss by timestamp
        policy_loss = torch.sum(-torch.sum(batch_policy * torch.log(
            policy_pred + EPS), dim=1, keepdim=True) * batch_timestamp) / torch.sum(batch_timestamp)

        # Weighted MSE loss by timestamp
        value_loss = torch.sum(
            (batch_value - value_pred) ** 2 * batch_timestamp) / torch.sum(batch_timestamp)

        if train:
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_policy_loss += policy_loss.detach()
        total_value_loss += value_loss.detach()
        num_batches += 1

    average_policy_loss = total_policy_loss / num_batches
    average_value_loss = total_value_loss / num_batches

    # Use distributed all reduce to average the losses across all processes.
    if USE_DDP:
        torch.distributed.barrier()
        torch.distributed.all_reduce(average_policy_loss,
                                     op=dist.ReduceOp.SUM)
        torch.distributed.all_reduce(average_value_loss,
                                     op=dist.ReduceOp.SUM)
        torch.distributed.barrier()

        average_value_loss /= world_size
        average_policy_loss /= world_size

    return average_policy_loss, average_value_loss


def main():
    if USE_DDP:
        setup_DDP()
    # Create the necessary directories
    os.makedirs(f"data/{RUN_NAME}/games", exist_ok=True)
    os.makedirs(f"data/{RUN_NAME}/models", exist_ok=True)

    logger.info(f"Created necessary directories for {RUN_NAME}.")

    network = ProverLLM(
        heads_only=True
    )
    if USE_DDP:
        network.to(local_rank)
    else:
        network.to(non_ddp_device)

    # load the previous model if it exists
    start_iteration = 0
    for i in range(NUM_ITERS):
        if os.path.exists(f"./data/{RUN_NAME}/models/{RUN_NAME}_iteration_{i}.pt"):
            start_iteration = i + 1

    if start_iteration > 0:
        state_dict = torch.load(
            f"./data/{RUN_NAME}/models/{RUN_NAME}_iteration_{start_iteration - 1}.pt")
        network.load_policy_value_state_dict(state_dict)
        logger.info(f"Loaded model from iteration {start_iteration - 1}.")
    if USE_DDP:
        network = DDP(network, device_ids=[
                      local_rank], output_device=local_rank)
        logger.info(
            f"Local rank {local_rank} network is on device {network.device}")
    else:
        logger.info(f"Network is on device {non_ddp_device}.")

    optimizer = optim.Adam(network.parameters(), lr=LR_INIT)

    timing_filepath = f"data/{RUN_NAME}/timing.txt"
    os.makedirs(os.path.dirname(timing_filepath), exist_ok=True)
    if os.path.exists(timing_filepath):
        with open(timing_filepath, "r") as f:
            timestamps = [float(line.strip()) for line in f]
    else:
        timestamps = []

    muncher = data_muncher.DataMuncher(local_rank, rank, world_size,
                                       iter=start_iteration,
                                       **data_muncher_kwargs)

    for iteration in range(start_iteration, NUM_ITERS):
        start_time = time.time()
        logger.info(f"Starting iteration {iteration}...")

        train_dataset = muncher.get()
        train_network(network, optimizer,
                      iteration, **train_dataset)

        timestamps.append(time.time() - start_time)
        if local_rank == 0:
            with open(timing_filepath, "w") as f:
                f.write("\n".join(str(t) for t in timestamps))

    torch.distributed.barrier()
    logger.info(f"Rank {local_rank} finished training.")
    cleanup()


if __name__ == "__main__":
    main()
