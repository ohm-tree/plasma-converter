import multiprocessing
import time

from src.completion_worker import main as completion_process  # lean_worker entry point
from src.lean_worker import main as lean_process  # lean_worker entry point
from src.mcts_worker import main as worker_process  # lean_worker entry point
from src.policy_value_worker import context_main as context_process
from src.policy_value_worker import policy_value_main as policy_value_process

# todo: make this a config file.
distributed_config = {
    'num_worker_procs': 4,
    'num_completion_procs': 1,
    'num_context_procs': 1,
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
    context_queue = multiprocessing.Queue()
    policy_value_queue = multiprocessing.Queue()

    # There is one global queue for lean repl queries, because such queries are not batched.
    lean_queue = multiprocessing.Queue()

    # There is one global queue for master queries. These are used to signal to workers that
    # they should terminate.
    master_queue = multiprocessing.Queue()

    # Create worker processes
    worker_procs = [multiprocessing.Process(target=worker_process, kwargs={
        'task_id': i,
        'num_tasks': distributed_config['num_worker_procs'],
        'json_name': json_name,
        'queue': worker_queues[i],
        'completion_queue': completion_queue,
        'context_queue': context_queue,
        'lean_queue': lean_queue,
    }
    ) for i in range(distributed_config['num_worker_procs'])]

    completion_procs = [multiprocessing.Process(target=completion_process, kwargs={
        'completion_worker_id': i,
        'num_completion_workers': distributed_config['num_completion_procs'],
        'json_name': json_name,
        'gpu_set': [2 * i, 2 * i + 1],
        'master_queue': completion_queue,
        'worker_queues': worker_queues,
        'completion_queue': completion_queue,
        'completion_batch_size': 100,
        'custom_eos': ["\n", "```"]
    }
    )
        for i in range(distributed_config['num_completion_procs'])]

    gpu_offset = 2 * distributed_config['num_completion_procs']

    policy_value_procs = [multiprocessing.Process(target=policy_value_process, kwargs={
        'policy_value_worker_id': i,
        'num_policy_value_workers': distributed_config['num_policy_value_procs'],
        'json_name': json_name,
        'gpu_set': [gpu_offset + 2 * i, gpu_offset + 2 * i + 1],
        'master_queue': policy_value_queue,
        'worker_queues': worker_queues,
        'context_queue': context_queue,
        'policy_value_queue': policy_value_queue,
        'policy_value_batch_size': 100
    })
        for i in range(distributed_config['num_policy_value_procs'])]

    gpu_offset += 2 * distributed_config['num_policy_value_procs']

    context_procs = [multiprocessing.Process(target=context_process, kwargs={
        'context_worker_id': i,
        'num_context_workers': distributed_config['num_context_procs'],
        'json_name': json_name,
        'gpu_set': [gpu_offset + 2 * i, gpu_offset + 2 * i + 1],
        'master_queue': context_queue,
        'context_queue': context_queue,
        'policy_value_queue': policy_value_queue,
        'context_batch_size': 100
    })
        for i in range(distributed_config['num_context_procs'])]

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

    # Wait for one hour then terminate all
    time.sleep(3600)
    for w in worker_procs + completion_procs + policy_value_procs + lean_procs:
        w.terminate()

    print("All processes have been terminated.")
