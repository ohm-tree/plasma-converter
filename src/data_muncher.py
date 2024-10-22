"""data_muncher.py"""

import logging
import os
import time
from typing import Set, Tuple, list

import numpy as np
import torch

from src.games.lean_game import LeanState
from src.networks.prover_llm import ProverLLM

logger = logging.getLogger()


def show_memory(local_rank):
    t = torch.cuda.get_device_properties(local_rank).total_memory
    r = torch.cuda.memory_reserved(local_rank)
    a = torch.cuda.memory_allocated(local_rank)
    f = r - a  # Free inside reserved.
    logger.info(
        f"local_rank {local_rank} Total (MiB): {t / (2 ** 20)}, Reserved: {r / (2 ** 20)}, Allocated: {a / (2 ** 20)}, Free: {f / (2 ** 20)}")


class DataMuncher():
    def __init__(
        self,
        local_rank: int,
        rank: int,
        world_size: int,
        iter: int,
        num_worker_tasks: int,
        num_groups: int,
        run_name: str,
        use_linear_wgt: bool,
        worker_ttk: int,
        sync: bool,
        num_train_samples: int,
        num_save_samples: int,
        completion_model: ProverLLM
    ):
        logger.info(f"DataMuncher initialized with")
        logger.info(f"local_rank: {local_rank}")
        logger.info(f"rank: {rank}")
        logger.info(f"world_size: {world_size}")
        logger.info(f"iter: {iter}")
        logger.info(f"num_worker_tasks: {num_worker_tasks}")
        logger.info(f"num_groups: {num_groups}")
        logger.info(f"run_name: {run_name}")
        logger.info(f"use_linear_wgt: {use_linear_wgt}")
        logger.info(f"worker_ttk: {worker_ttk}")
        logger.info(f"sync: {sync}")
        logger.info(f"num_train_samples: {num_train_samples}")
        logger.info(f"num_save_samples: {num_save_samples}")
        logger.info(f"completion_model: {completion_model}")

        if world_size == 1:
            logger.warning(
                "Running in single process mode, setting local rank to torch device.")
            self.local_rank = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        else:
            self.local_rank: int = local_rank
        self.rank: int = rank
        self.world_size: int = world_size
        self.num_worker_tasks: int = num_worker_tasks

        self.my_workers: list[int] = list(
            range(rank, num_worker_tasks, world_size))
        self.live_workers: Set[int] = set(self.my_workers)
        self.worker_seen: list[int] = [0 for _ in range(num_worker_tasks)]

        self.num_groups: int = num_groups
        self.run_name: str = run_name

        self.use_linear_wgt: bool = use_linear_wgt
        self.worker_ttk: int = worker_ttk

        self.all_state_tensors: list[torch.Tensor] = []
        self.all_distr_tensors: list[torch.Tensor] = []
        self.all_outco_tensors: list[torch.Tensor] = []
        self.all_tmstp_tensors: list[torch.Tensor] = []

        self.sync: bool = sync
        self.iter: int = iter
        self.num_train_samples: int = num_train_samples
        self.num_save_samples: int = num_save_samples

        self.async_setup_called = False

        self.completion_model = completion_model

    def load_states(self, state_path):
        states: list[LeanState] = np.load(state_path)
        state_representations: list[torch.Tensor] = []
        for state in states:
            prompt = self.completion_model.value_policy_prompter(
                state)
            input_ids, attention_mask = self.completion_model.tokenize(
                prompt)
            intermediate_output = self.completion_model.get_intermediate_state(
                input_ids, attention_mask)

            state_representations.append(
                intermediate_output.detach())

        return torch.Tensor(state_representations)

    def sync_collate(self) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Synchronously collate data for this iteration from all workers."""

        new_states: list[torch.Tensor] = []
        new_distrs: list[torch.Tensor] = []
        new_outcos: list[torch.Tensor] = []
        new_tmstps: list[torch.Tensor] = []

        start_time = None

        finished_workers = set()

        # Loop until all workers have finished.
        while True:
            for task_id in self.live_workers:
                group = task_id // (self.num_worker_tasks // self.num_groups)

                thread_save_path = os.path.join(
                    "data", self.run_name, "games", f"{group}", f"{task_id}")

                state_path = os.path.join(
                    thread_save_path, f"{self.iter}_states.npy")
                distr_path = os.path.join(
                    thread_save_path, f"{self.iter}_distributions.npy")
                outco_path = os.path.join(
                    thread_save_path, f"{self.iter}_outcomes.npy")

                if (not task_id in finished_workers and all(os.path.exists(p) for p in [state_path, distr_path, outco_path])):
                    # Worker is unfinished and has finished collecting data.
                    finished_workers.add(task_id)
                    states = self.load_states(state_path)
                    distrs = torch.Tensor(np.load(distr_path))
                    outcos = torch.Tensor(np.load(outco_path))
                    tmstps = torch.Tensor(
                        [self.iter + 1 if self.use_linear_wgt else 1 for _ in range(states.shape[0])])

                    assert states.shape[0] == distrs.shape[0] == outcos.shape[0] == tmstps.shape[0]

                    new_states.append(states)
                    new_distrs.append(distrs)
                    new_outcos.append(outcos)
                    new_tmstps.append(tmstps)

            if start_time is None and len(self.live_workers) > len(finished_workers) > len(self.live_workers) // 2:
                # Start the kill timer.
                logger.info(
                    "Over half of the workers have finished. Starting timer to kill the rest.")
                start_time = time.time()

            if start_time is not None and time.time() - start_time > self.worker_ttk:
                # Kill timer began and has expired, kill the rest of the workers and exit.
                for task_id in self.live_workers - finished_workers:
                    self.live_workers.remove(task_id)

                logger.info(
                    f"Killing unfinished workers: {len(self.live_workers)} left.")
                break

            if len(finished_workers) == len(self.live_workers):
                # All workers have finished, exit.
                break

            # Continue spinning on data from workers.
            logger.info(
                f"Spinning on workers to finish... {len(finished_workers)} / {len(self.live_workers)} are complete.")
            time.sleep(30)

        logger.info(
            f"Total samples for iteration {self.iter}: {sum(s.shape[0] for s in new_states)}")
        return new_states, new_distrs, new_outcos, new_tmstps

    def async_setup(self):
        self.async_setup_called = True
        # In the special case where iteration is 0,
        # we spin until every single worker has data,
        # including the workers that belong to other controllers.
        while True:
            all_workers_have_data = True
            for task_id in range(self.num_worker_tasks):
                group = task_id // (self.num_worker_tasks // self.num_groups)

                thread_save_path = os.path.join(
                    "data", self.run_name, "games", f"{group}", f"{task_id}")
                print(thread_save_path)

                states_path = os.path.join(
                    thread_save_path, f"{self.worker_seen[task_id]}_states.npy")
                distrs_path = os.path.join(
                    thread_save_path, f"{self.worker_seen[task_id]}_distributions.npy")
                outcos_path = os.path.join(
                    thread_save_path, f"{self.worker_seen[task_id]}_outcomes.npy")

                if not all(os.path.exists(p) for p in [states_path, distrs_path, outcos_path]):
                    all_workers_have_data = False
                    logger.info(
                        f"Waiting for all workers to have data (missing {task_id})")
                    break

            if all_workers_have_data:
                break

            time.sleep(10)

        # In the special case where iteration is 0,
        # We might be bootstrapping! In this case, we definitely do not want to load in
        # *all* previous games that we have seen. We will only load in the last ~873 games
        # per worker, based on the empirically observed 200 samples per game.
        num_wanted = float("inf")
        # int(self.num_save_samples /
        #                  (200 * self.num_worker_tasks / self.world_size))
        logger.info(
            f"If bootstrapping: only loading in the last {num_wanted} games per worker.")

        # To this end, for each worker, we check how many games exist, and then
        # set self.worker_seen[task_id] to the maximum of that number and the number of games we want.
        for task_id in range(self.num_worker_tasks):
            group = task_id // (self.num_worker_tasks // self.num_groups)

            thread_save_path = os.path.join(
                "data", self.run_name, "games", f"{group}", f"{task_id}")
            files = os.listdir(thread_save_path)
            num_games = len(files) // 3
            self.worker_seen[task_id] = max(num_games - num_wanted, 0)
            logger.info(
                f"Worker {task_id} has {num_games} games, setting worker_seen to {self.worker_seen[task_id]}")

    def async_collate(self) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """For each worker, scoop up all data that has not already been scooped up."""
        if not self.async_setup_called:
            self.async_setup()
        new_states: list[torch.Tensor] = []
        new_distrs: list[torch.Tensor] = []
        new_outcos: list[torch.Tensor] = []
        new_tmstps: list[torch.Tensor] = []

        for task_id in self.my_workers:
            group = task_id // (self.num_worker_tasks // self.num_groups)

            thread_save_path = os.path.join(
                "data", self.run_name, "games", f"{group}", f"{task_id}")
            files = os.listdir(thread_save_path)
            num_games = len(files) // 3
            for i in range(self.worker_seen[task_id], num_games):
                states_path = os.path.join(
                    thread_save_path, f"{i}_states.npy")
                distrs_path = os.path.join(
                    thread_save_path, f"{i}_distributions.npy")
                outcos_path = os.path.join(
                    thread_save_path, f"{i}_outcomes.npy")

                states = self.load_states(states_path)
                distrs = torch.Tensor(np.load(distrs_path))
                outcos = torch.Tensor(np.load(outcos_path))
                tmstps = torch.Tensor(
                    [self.iter + 1 if self.use_linear_wgt else 1 for _ in range(states.shape[0])])

                assert states.shape[0] == distrs.shape[0] == outcos.shape[0] == tmstps.shape[0]

                new_states.append(states)
                new_distrs.append(distrs)
                new_outcos.append(outcos)
                new_tmstps.append(tmstps)

            if self.iter == 0:
                # We might be bootstrapping, in which case this will take a long time and it's good to have debug.
                logger.info(f"Finished scooping data from worker {task_id}")

            self.worker_seen[task_id] = num_games
        logger.info(
            f"Total new samples for iteration {self.iter}: {sum(s.shape[0] for s in new_states)}")
        logger.info(f"Total scooped samples from each worker:")
        logger.info(" ".join(str(i) for i in self.worker_seen))

        return new_states, new_distrs, new_outcos, new_tmstps

    def get(self):
        if self.sync:
            new_states, new_distrs, new_outcos, new_tmstps = self.sync_collate()
        else:
            new_states, new_distrs, new_outcos, new_tmstps = self.async_collate()

        # Do not push anybody to GPU yet, before we clear out the old data from VRAM.

        self.all_state_tensors.extend(new_states)
        self.all_distr_tensors.extend(new_distrs)
        self.all_outco_tensors.extend(new_outcos)
        self.all_tmstp_tensors.extend(new_tmstps)

        logger.info(
            f"Before truncating, the total number of games is: {len(self.all_state_tensors)}")
        logger.info(
            f"Before truncating, the total number of samples is: {sum(s.shape[0] for s in self.all_state_tensors)}")

        assert len(self.all_state_tensors) == len(self.all_distr_tensors) == len(
            self.all_outco_tensors) == len(self.all_tmstp_tensors)

        num_samples = sum(s.shape[0] for s in self.all_state_tensors)

        while num_samples > self.num_save_samples:
            num_samples -= self.all_state_tensors[0].shape[0]
            del self.all_state_tensors[0]
            del self.all_distr_tensors[0]
            del self.all_outco_tensors[0]
            del self.all_tmstp_tensors[0]

        logger.info(
            f"After truncating, the total number of games is: {len(self.all_state_tensors)}")
        logger.info(
            f"After truncating, the total number of samples is: {sum(s.shape[0] for s in self.all_state_tensors)}")

        # Our objective here is to get a subset of the data
        # that contains exactly num_train_samples samples.
        # We first over-shoot, then we truncate.
        index_permutation = np.random.permutation(len(self.all_state_tensors))
        train_cutoff = 0
        train_samples = 0
        while train_samples < self.num_train_samples and train_cutoff < len(index_permutation):
            train_samples += self.all_state_tensors[index_permutation[train_cutoff]].shape[0]
            train_cutoff += 1

        logger.info(f"Total number of games used for training: {train_cutoff}")
        logger.info(
            f"Total number of samples used for training: {train_samples}")

        index_subset = index_permutation[:train_cutoff]
        train_state_tensor = torch.cat(
            [self.all_state_tensors[i] for i in index_subset], dim=0).to(self.local_rank)
        train_distr_tensor = torch.cat(
            [self.all_distr_tensors[i] for i in index_subset], dim=0).to(self.local_rank)
        train_outco_tensor = torch.cat(
            [self.all_outco_tensors[i] for i in index_subset], dim=0).to(self.local_rank)
        train_tmstp_tensor = torch.cat(
            [self.all_tmstp_tensors[i] for i in index_subset], dim=0).to(self.local_rank)

        train_tmstp_tensor = train_tmstp_tensor - \
            torch.min(train_tmstp_tensor) + 1
        # show_memory(self.local_rank)
        self.iter += 1

        logger.info("Pushed data to GPU.")
        return {
            "state_tensor": train_state_tensor,
            "distribution_tensor": train_distr_tensor,
            "outcome_tensor": train_outco_tensor,
            "timestamp_tensor": train_tmstp_tensor
        }
