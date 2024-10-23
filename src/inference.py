"""
This file runs inference with a yaml config.
"""

import argparse
import multiprocessing
import time

import yaml

from src.collate_solutions import collate_results, collate_solutions
from src.workers import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)


"""
Run fast linear inference with
```bash
python src/inference.py --config configs/linear.yaml
```

Debug fast linear inference with
```bash
python src/inference.py --config configs/linear_debug.yaml
```

Run fast mcts inference with
```bash
python src/inference.py --config configs/fast_mcts.yaml
```

Debug fast mcts inference with
```bash
python src/inference.py --config configs/fast_mcts_debug.yaml
```


"""


def run_inference():
    run_name = config['run_name'] + time.strftime("_%Y-%m-%d_%H-%M-%S")

    # save a copy of the config in the results folder
    os.makedirs(os.path.join('results', run_name), exist_ok=True)
    with open(os.path.join('results', run_name, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    for type_string, _, _, _ in WORKER_TYPES_AND_STRINGS:
        if type_string not in config:
            config[type_string] = {'num_procs': 0}

    queues: dict[str, multiprocessing.Queue] = {}
    print("Creating Queues:")
    for type_string, _, _, _ in WORKER_TYPES_AND_STRINGS:
        queues.update(
            {
                type_string + "_" + str(i): multiprocessing.Queue()
                for i in range(config[type_string]['num_procs'])
            }
        )
        print(f"{type_string}: {config[type_string]['num_procs']} queues")

    for type_string, _, _, _ in WORKER_TYPES_AND_STRINGS:
        if config[type_string]['num_procs'] == 0:
            continue
        queues.update({type_string: multiprocessing.Queue()})
        print(f"{type_string}: 1 queues")

    # Create inference processes
    gpu_offset = 0
    procs: dict[str, list[multiprocessing.Process]] = {}
    for type_string, entrypoint, gpu, _ in WORKER_TYPES_AND_STRINGS:
        procs.update({type_string: []})
        for i in range(config[type_string]['num_procs']):
            procs[type_string].append(multiprocessing.Process(target=entrypoint, kwargs={
                'global_config': config,
                'config': config[type_string],
                'run_name': run_name,
                'worker_type': type_string,
                'task_id': i,
                'queues': queues,
                'gpu_set': [gpu_offset + i] if gpu else []}
            )
            )

        if gpu:
            gpu_offset += config[type_string]['num_procs']

    print("Starting Processes:")
    for key in procs.keys():
        print(f"{key}: {len(procs[key])} processes")

    time.sleep(1)  # Debug: let us see the processes start up

    # These should all just be references, arranged in nice ways.
    all_procs: list[multiprocessing.Process] = []
    for proc_type, proc_list in procs.items():
        for w in proc_list:
            all_procs.append(w)

    # Start all processes
    for w in all_procs:
        w.start()

    alive = {key: [True for i in procs[key]] for key in procs.keys()}

    def print_alive():
        print("Still alive:")
        for key in procs.keys():
            print(f"{key}: {[i for i, v in enumerate(alive[key]) if v]}")

    while True:
        # Check if all the processes are still alive
        for key in procs.keys():
            for i, w in enumerate(procs[key]):
                w.join(timeout=0)
                if not w.is_alive():
                    if alive[key][i]:
                        print(f"{key} process number {i} has died.")
                        print_alive()
                    alive[key][i] = False

        # We terminate if all of the inessential workers are dead
        # or if any of the essential workers are dead
        all_inessential_dead = True
        any_essential_dead = False
        for type_string, _, _, inessential in WORKER_TYPES_AND_STRINGS:
            for i in range(config[type_string]['num_procs']):
                if inessential:
                    all_inessential_dead = all_inessential_dead and not alive[type_string][i]
                else:
                    any_essential_dead = any_essential_dead or not alive[type_string][i]

        if any_essential_dead:
            print("An essential worker has died.")
            break
        if all_inessential_dead:
            print("All inessential workers are done.")
            break
        time.sleep(1)

    print("Killing all processes.")
    # Send kill signals to every queue.
    for key in queues.keys():
        queues[key].put('kill')

    print("All kill signals sent.")

    # Wait for 30 seconds, then force kill all processes
    time.sleep(30)
    for w in all_procs:
        w.terminate()
        w.join()
    print("All processes terminated.")

    # Collate results
    collate_results()
    collate_solutions()


if __name__ == '__main__':
    run_inference()
