"""
This script collates the results of each inference run and saves them to a file in the results folder.
"""
import json
import os

from src.lean.lean_game import LeanGame, LeanState


def load_problems():
    # I live in src/workers/
    WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(WORKER_DIR)
    ROOT_DIR = os.path.dirname(SRC_DIR)
    data = {}

    with open('datasets/minif2f.jsonl', 'r') as file:
        for line in file.readlines():
            problem = json.loads(line.strip())
            informal_prefix = problem['informal_prefix']
            formal_statement = problem['formal_statement']
            PROBLEM_STATEMENT = informal_prefix + formal_statement
            tactic_state = problem['goal']

            game: LeanGame = LeanGame(
                worker=None,
                problem=PROBLEM_STATEMENT,
                tactic_state=tactic_state,
                max_depth=40
            )

            data.update({problem['name']: game})
    return data


def collate_solutions():

    problem_data = load_problems()

    all_runs = os.listdir('results')
    all_results = {}
    print(all_runs)
    for run in all_runs:
        if not os.path.isdir(os.path.join('results', run)):
            continue
        all_results[run] = []

        # load the results
        for problem_path in os.listdir(os.path.join('results', run)):
            if problem_path == "config.yaml":  # Skip config files
                continue
            result_dict = {}
            # Problem: aime_1983_p1
            # Split: test
            # Result: -1.0
            with open(os.path.join('results', run, problem_path), 'r') as file:
                problem = file.readline().strip()
                split = file.readline().strip()
                result = file.readline().strip()
                try:
                    result_dict = {
                        'problem': problem.split(': ')[1],
                        'split': split.split(': ')[1],
                        'result': result.split(': ')[1]
                    }
                except:
                    print(f"Error parsing {run}/{problem_path}")
                    continue
            with open(os.path.join('outputs', run, problem_path), 'r') as file:
                # LeanState(code='  norm_num [add_assoc, add_comm, add_left_comm] at h₀ h₁ h₂ h₃\n  ring_nf at h₀ h₁ h₂ h₃\n  linarith\n', depth=3, tactic_state='', dead=False)
                # Suspicious code, just use eval() to create the LeanState object
                code = file.readlines()[-1]  # Last line
                state = eval(code)  # Dangerous, but we trust the output
                result_dict['state'] = state
            all_results[run].append(result_dict)

    for run in all_results:
        with open(os.path.join('results', run + '.lean'), 'w') as file:
            for problem in all_results[run]:
                file.write(f"-" * 80 + "\n")
                file.write(f"-" * 80 + "\n")
                file.write(problem_data[problem['problem']].pretty_print(
                    problem['state']) + "\n")

                file.write(f"-- Problem: {problem['problem']}\n")
                file.write(f"-- Split: {problem['split']}\n")
                file.write(f"-- Result: {problem['result']}\n")
                file.write("\n")


def collate_results():
    # look in the results folder for all the results
    # collate them into a single file
    # save the file in the results folder

    all_runs = os.listdir('results')
    all_results = {}
    print(all_runs)
    for run in all_runs:
        if not os.path.isdir(os.path.join('results', run)):
            continue
        all_results[run] = []
        # load the results
        for problem_path in os.listdir(os.path.join('results', run)):
            if problem_path == "config.yaml":  # Skip config files
                continue
            result_dict = {}
            # Problem: aime_1983_p1
            # Split: test
            # Result: -1.0
            with open(os.path.join('results', run, problem_path), 'r') as file:
                problem = file.readline().strip()
                split = file.readline().strip()
                result = file.readline().strip()
                try:
                    result_dict = {
                        'problem': problem.split(': ')[1],
                        'split': split.split(': ')[1],
                        'result': result.split(': ')[1]
                    }
                except:
                    print(f"Error parsing {run}/{problem_path}")
                    continue
            all_results[run].append(result_dict)

    # aggregate the results.
    # for each run, count the number of problems solved in each split.

    num_solved = {}
    denominator = {}
    for run in all_results:
        num_solved[run] = {'valid': 0, 'test': 0}
        denominator[run] = {'valid': 0, 'test': 0}
        for result in all_results[run]:
            if result['result'] == '1.0':
                num_solved[run][result['split']] += 1
            denominator[run][result['split']] += 1

    # save the results in a file
    with open('results/results.txt', 'w') as file:
        for run in num_solved:
            file.write(run + '\n')
            file.write(
                f"Valid: {num_solved[run]['valid']}/{denominator[run]['valid']}\n")
            file.write(
                f"Test: {num_solved[run]['test']}/{denominator[run]['test']}\n")


if __name__ == '__main__':
    collate_results()
    collate_solutions()
