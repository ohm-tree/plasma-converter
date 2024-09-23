import multiprocessing
import time

from src.completion_worker import main as completion_process  # lean_worker entry point
from src.lean_worker import main as lean_process  # lean_worker entry point
from src.mcts_worker import main as worker_process  # lean_worker entry point
from src.policy_value_worker import (
    main as policy_value_process,  # lean_worker entry point
)

# todo: make this a config file.
distributed_config = {
    'num_worker_procs': 4,
    'num_completion_procs': 1,
    'num_policy_value_procs': 1,
    'num_lean_procs': 1,
}
json_name = "config"  # todo: make this a config file.

if __name__ == "__main__":
    worker_queues = {i: multiprocessing.Queue()
                     for i in range(distributed_config['num_worker_procs'])}

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
    completion_queue = multiprocessing.Queue()
    policy_value_queue = multiprocessing.Queue()

    # There is one global queue for lean repl queries, because such queries are not batched.
    lean_queue = multiprocessing.Queue()

    # There is one global queue for master queries. These are used to signal to workers that
    # they should terminate.
    master_queue = multiprocessing.Queue()

    # Create worker processes
    worker_procs = [multiprocessing.Process(target=worker_process, kwargs={
        'queue': worker_queues[i],
        'completion_queue': completion_queue,
        'policy_value_queue': policy_value_queue,
        'lean_queue': lean_queue,
        'task_id': i,
        'num_tasks': distributed_config['num_worker_procs'],
        'json_name': json_name
    }
    ) for i in range(distributed_config['num_worker_procs'])]

    completion_procs = [multiprocessing.Process(target=completion_process, kwargs={
        'task_id': i,
        'num_tasks': distributed_config['num_completion_procs'],
        'json_name': json_name,
        'master_queue': completion_queue,
        'worker_queues': worker_queues,
        'completion_queue': completion_queue,
        'completion_batch_size': 100
    }
    )
        for i in range(distributed_config['num_completion_procs'])]

    policy_value_procs = [multiprocessing.Process(target=policy_value_process, args=(policy_value_queue,))
                          for _ in range(distributed_config['num_policy_value_procs'])]

    lean_procs = [multiprocessing.Process(target=lean_process, kwargs={
        'task_id': i,
        'num_tasks': distributed_config['num_lean_procs'],
        'json_name': json_name,
        'master_queue': lean_queue,
        'worker_queues': worker_queues,
        'lean_queue': lean_queue
    }
    )
        for i in range(distributed_config['num_lean_procs'])]

    # Start all processes
    for w in worker_procs + completion_procs + policy_value_procs + lean_procs:
        w.start()
