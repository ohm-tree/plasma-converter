"""
This file runs inference with a yaml config.
"""

import argparse
import multiprocessing
import time
from multiprocessing import sharedctypes

import yaml

from src.collate_solutions import collate_results, collate_solutions
from src.workers import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('-y', action='store_true')

args = parser.parse_args()


"""
Run fast linear inference with
```bash
python src/inference.py --config configs/linear.yaml
```

Debug fast linear inference with
```bash
python src/inference.py --config configs/linear_debug.yaml
```

Run full mcts inference with
```bash
python src/inference.py --config configs/full_mcts.yaml
```

Run fast mcts inference with
```bash
python src/inference.py --config configs/fast_mcts.yaml
```

Pure mcts inference with
```bash
python src/inference.py --config configs/fast_mcts_pure.yaml
```

Debug fast mcts inference with
```bash
python src/inference.py --config configs/fast_mcts_debug.yaml
```


"""


def config_load_dialog() -> tuple[dict, str]:
    """
    Load the configuration from the provided YAML file and handle existing runs.

    This function checks for existing runs with the same configuration and prompts the user
    to either start a new run or resume an existing one. It also saves the configuration
    for the current run.

    Returns:
    -------
    config : dict
        The loaded configuration dictionary.
    run_name : str
        The name of the current run.
    """

    # first, read the config of every folder in the 'results' folder.
    # If any of them are the same, then append to matches.
    # 1. Warn the user, ask them if they would like to start a new run (default) or resume a run (they should select from the matches).
    # 2. If they choose to resume, simply set the run name to that folder's name. add a new file called restarts.txt to that folder and append the current time to it.
    # 3. If they choose to start a new run, create a new folder with a timestamp.

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    run_name = config['run_name'] + time.strftime("_%Y-%m-%d_%H-%M-%S")

    # check for the -y flag
    if args.y:
        print(f"Starting new run: {run_name}")
    else:
        existing_runs = os.listdir('results')
        matches = []
        for run in existing_runs:
            if not os.path.isdir(os.path.join('results', run)):
                continue
            if run.startswith(config['run_name']):
                # possible match. read the yaml file.
                with open(os.path.join('results', run, 'config.yaml'), 'r') as file:
                    existing_config = yaml.safe_load(file)
                    if existing_config == config:
                        matches.append(run)  # add to matches

        if matches:
            print("Found existing runs that match the current configuration:")
            for i, match in enumerate(matches):
                print(f"{i + 1}: {match}")
            choice = input("Would you like to start a new run (y/n)? ")
            if choice.lower() == 'n':
                choice = input("Which run would you like to resume? ")
                run_name = matches[int(choice) - 1]
                print(f"Resuming run: {run_name}")

                with open(os.path.join('results', run_name, 'restarts.txt'), 'a') as file:
                    # append the current time
                    file.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            else:
                print(f"Starting new run: {run_name}")
        else:
            print(f"Starting new run: {run_name}")

    # save a copy of the config in the results folder
    os.makedirs(os.path.join('results', run_name), exist_ok=True)
    with open(os.path.join('results', run_name, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    return config, run_name


def run_inference():
    config, run_name = config_load_dialog()

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
        print(
            f"{type_string} worker-specific: {config[type_string]['num_procs']} queues")

    for type_string, _, _, _ in WORKER_TYPES_AND_STRINGS:
        if config[type_string]['num_procs'] == 0:
            continue
        queues.update({type_string: multiprocessing.Queue()})
        print(f"{type_string}: 1 queues")

    values: dict[str, sharedctypes.Synchronized] = {}

    values['problem_number'] = sharedctypes.Value('i', 0)

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
                'values': values,
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
