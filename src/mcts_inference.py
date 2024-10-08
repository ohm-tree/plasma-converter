"""
This file runs mcts inference.
"""

import argparse
import multiprocessing
import time
import yaml

from src.workers.completion_worker import main as completion_process
from src.workers.lean_worker import main as lean_process  # lean_worker entry point
from src.workers.mcts_inference_worker import main as inference_process
from src.workers.policy_value_worker import context_main as context_process
from src.workers.policy_value_worker import policy_value_main as slow_pv
from src.workers.fast_pv_worker import policy_value_main as fast_pv


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)


# Run with `python src/mcts_inference.py --config configs/basic.yaml`
def run_inference():
    policy_value_process = fast_pv if config['policy_value']['type'] == 'fast' else slow_pv

    run_name = config['run_name'] + time.strftime("_%Y-%m-%d_%H-%M-%S")

    # task-specific queues
    worker_queues = {i: multiprocessing.Queue()
                     for i in range(config['worker']['num_procs'])}
    completion_queues = {i: multiprocessing.Queue()
                         for i in range(config['completion']['num_procs'])}
    policy_value_queues = {i: multiprocessing.Queue()
                           for i in range(config['policy_value']['num_procs'])}
    context_queues = {i: multiprocessing.Queue()
                      for i in range(config['context']['num_procs'])}
    lean_queues = {i: multiprocessing.Queue()
                   for i in range(config['lean']['num_procs'])}

    # Policy_value and completion queues are batched. One idea is to assign a many-to-one mapping
    # between worker processes and completion/policy_value processes, such that each
    # completion/policy_value process is responsible for a subset of worker processes.

    # Instead, we have global queues for each.
    # Both the completion and policy_value processes use the following algorithm:
    # While True:
    #   1. Check for kill signals from the master queue.
    #   2. Collect new tasks from the completion/policy_value queue until either there are no tasks
    #      left on the queue (a timeout occurs) or we have collected COMPLETION_BATCH_SIZE or
    #      POLICY_VALUE_BATCH_SIZE tasks.
    global_completion_queue = multiprocessing.Queue()
    global_context_queue = multiprocessing.Queue()
    global_policy_value_queue = multiprocessing.Queue()

    # There is one global queue for lean repl queries, because such queries are not batched.
    global_lean_queue = multiprocessing.Queue()

    # There is one global queue for master queries. It is used to tell the master
    # process that you are finished working.
    master_queue = multiprocessing.Queue()

    # Create inference processes
    inference_procs = [multiprocessing.Process(target=inference_process, kwargs={
        'config': config,
        'run_name': run_name,
        'task_id': i,
        'master_queue': master_queue,
        'worker_queue': worker_queues[i],
        'global_completion_queue': global_completion_queue,
        'global_lean_queue': global_lean_queue,
        'global_context_queue': global_context_queue,
    }
    ) for i in range(config['worker']['num_procs'])]

    completion_procs = [multiprocessing.Process(target=completion_process, kwargs={
        'config': config['completion'],
        'run_name': run_name,
        'completion_worker_id': i,
        'gpu_set': [i],
        'master_queue': master_queue,
        'completion_queue': completion_queues[i],
        'worker_queues': worker_queues,
        'global_completion_queue': global_completion_queue,
    }
    )
        for i in range(config['completion']['num_procs'])]

    gpu_offset = config['completion']['num_procs']

    policy_value_procs = [multiprocessing.Process(target=policy_value_process, kwargs={
        'config': config['policy_value'],
        'run_name': run_name,
        'policy_value_worker_id': i,
        'gpu_set': [gpu_offset + i],
        'master_queue': master_queue,
        'policy_value_queue': policy_value_queues[i],
        'worker_queues': worker_queues,
        'global_policy_value_queue': global_policy_value_queue,
    })
        for i in range(config['policy_value']['num_procs'])]

    gpu_offset += config['policy_value']['num_procs']

    # context_procs = [multiprocessing.Process(target=context_process, kwargs={
    #     'config': config['context'],
    #     'run_name': run_name,
    #     'context_worker_id': i,
    #     'gpu_set': [gpu_offset + i],
    #     'master_queue': master_queue,
    #     'context_queue': context_queues[i],
    #     'global_context_queue': global_context_queue,
    #     'global_policy_value_queue': global_policy_value_queue,
    # })
    #     for i in range(config['context']['num_procs'])]

    lean_procs = [multiprocessing.Process(target=lean_process, kwargs={
        'config': config['lean'],
        'run_name': run_name,
        'task_id': i,
        'master_queue': master_queue,
        'lean_queue': lean_queues[i],
        'worker_queues': worker_queues,
        'global_lean_queue': global_lean_queue
    }
    )
        for i in range(config['lean']['num_procs'])]

    # Start all processes
    for w in inference_procs + completion_procs + lean_procs + policy_value_procs:
        w.start()

    workers_alive = {i: True for i in range(
        config['worker']['num_procs'])}
    completion_alive = {i: True for i in range(
        config['completion']['num_procs'])}
    lean_alive = {i: True for i in range(config['lean']['num_procs'])}
    policy_value_alive = {i: True for i in range(
        config['policy_value']['num_procs'])}
    context_alive = {i: True for i in range(
        config['context']['num_procs'])}

    while True:
        # check if all the processes are still alive
        for i, w in enumerate(inference_procs):
            w.join(timeout=0)
            if not w.is_alive():
                workers_alive[i] = False
        for i, w in enumerate(completion_procs):
            w.join(timeout=0)
            if not w.is_alive():
                completion_alive[i] = False
        for i, w in enumerate(lean_procs):
            w.join(timeout=0)
            if not w.is_alive():
                lean_alive[i] = False
        for i, w in enumerate(policy_value_procs):
            w.join(timeout=0)
            if not w.is_alive():
                policy_value_alive[i] = False
        for i, w in enumerate(context_procs):
            w.join(timeout=0)
            if not w.is_alive():
                context_alive[i] = False

        if all([not v for v in workers_alive.values()]):
            print("All workers are done.")
            break
        if any([not v for v in completion_alive.values()]):
            print("One of the completion processes has died.")
            break
        if any([not v for v in lean_alive.values()]):
            print("One of the lean processes has died.")
            break
        if any([not v for v in policy_value_alive.values()]):
            print("One of the policy_value processes has died.")
            break
        if any([not v for v in context_alive.values()]):
            print("One of the context processes has died.")
            break

        time.sleep(1)

    # Send kill signals to all processes
    for q in worker_queues.values():
        q.put("kill")
    for q in completion_queues.values():
        q.put("kill")
    for q in lean_queues.values():
        q.put("kill")
    for q in policy_value_queues.values():
        q.put("kill")
    for q in context_queues.values():
        q.put("kill")
    print("All kill signals sent.")

    # Wait for 5 minutes, then force kill all processes
    time.sleep(300)
    for w in inference_procs + completion_procs + lean_procs + policy_value_procs:
        w.terminate()
        w.join()
    print("All processes terminated.")

run_inference()