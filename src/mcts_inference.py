"""
This file runs mcts inference with fast_pv_worker and a yaml config.
"""

import argparse
import multiprocessing
import time
from typing import Callable, Dict, List, Tuple

import yaml

from src.workers import *
from src.workers.worker import TaskType, WorkerIdentifer, WorkerType

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)


# Run with `python src/mcts_inference_fast.py --config configs/basic.yaml`
def run_inference():
    run_name = config['run_name'] + time.strftime("_%Y-%m-%d_%H-%M-%S")

    # Assert that the config is valid
    fast_pv_active = (config['fast_policy_value']['num_procs'] > 0)
    pv_active = (config['policy_value']['num_procs'] >
                 0 or config['context']['num_procs'] > 0)
    if fast_pv_active:
        if pv_active:
            raise ValueError(
                "fast_policy_value and policy_value/context workers cannot be run at the same time.")
        else:
            print("Running in fast PV mode.")
    else:
        if pv_active:
            print("Running in normal PV mode.")
        else:
            print("Warning: No policy value workers are active.")

    WORKER_TYPES_AND_STRINGS: Tuple[Tuple[WorkerType, str, Callable, bool]] = (
        (MCTSWorkerType, 'mcts', mcts_inference_entrypoint, False),
        (LinearInferenceWorkerType, 'linear_inference',
         linear_inference_entrypoint, True),
        (CompletionWorkerType, 'completion', completion_entrypoint, True),
        (PolicyValueWorkerType, 'policy_value', policy_value_entrypoint, True),
        (LeanWorkerType, 'lean', lean_entrypoint, False),
        (FastPolicyValueWorkerType, 'fast_policy_value',
         fast_policy_value_entrypoint, True),
        (ContextWorkerType, 'context', context_entrypoint, True),
    )

    queues = {}
    for worker_type, type_string, _, _ in WORKER_TYPES_AND_STRINGS:
        queues.update(
            {
                WorkerIdentifer(worker_type, i): multiprocessing.Queue()
                for i in range(config[type_string]['num_procs'])
            }
        )

    for task_type in [LeanTaskType, CompletionTaskType, PolicyValueTaskType, PolicyValuePostProcessTaskType, KillTaskType]:
        queues.update({task_type: multiprocessing.Queue()})

    # Create inference processes
    gpu_offset = 0
    procs: Dict[str, List[multiprocessing.Process]] = {}
    for _, type_string, entrypoint, gpu in WORKER_TYPES_AND_STRINGS:
        if config[type_string]['num_procs'] > 0:
            procs.update({type_string: []})
            for i in range(config[type_string]['num_procs']):
                procs[type_string].append(multiprocessing.Process(target=entrypoint, kwargs={
                    'config': config[type_string],
                    'run_name': run_name,
                    'task_id': i,
                    'queues': queues,
                    'gpu_set': [gpu_offset + i] if gpu else []}
                )
                )

        if gpu:
            gpu_offset += config[type_string]['num_procs']

    # These should all just be references, arranged in nice ways.
    all_procs: List[multiprocessing.Process] = []
    for proc_type, proc_list in procs.items():
        for w in proc_list:
            all_procs.append(w)

    # Start all processes
    for w in all_procs:
        w.start()

    alive = {i: [True for i in procs[key]] for key in procs.keys()}

    while True:
        # Check if all the processes are still alive
        for key in procs.keys():
            for i, w in enumerate(procs[key]):
                w.join(timeout=0)
                if not w.is_alive():
                    alive[key][i] = False

        dead = False
        if all([not v for v in alive['linear_inference']]):
            print("All workers are done.")
            dead = True
            break

        for key in procs.keys():
            if key == 'linear_inference':
                continue
            if any([not v for v in alive[key]]):
                print(f"One of the {key} processes has died.")
                dead = True
                break
        if dead:
            break
        time.sleep(1)

    # Send kill signals to all processes
    for worker_type, type_string, _, _ in WORKER_TYPES_AND_STRINGS:
        for i in range(config[type_string]['num_procs']):
            queues[WorkerIdentifer(worker_type, i)].put("kill")
    print("All kill signals sent.")

    # Wait for 5 minutes, then force kill all processes
    time.sleep(300)
    for w in all_procs:
        w.terminate()
        w.join()
    print("All processes terminated.")


if __name__ == '__main__':
    run_inference()
