"""
In this file, we will let a human play a game of Lean (using modal).
"""

import json
import os
import time
from typing import Optional

import pexpect
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.games.leaner_lean_game import LeanGame, LeanGameState

# set "VLLM_LOGGING_LEVEL" to "WARNING" to suppress logging
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

HOME_DIR = os.path.expanduser('~')
print("HOME_DIR", HOME_DIR)

with open(f"{HOME_DIR}/plasma-converter/datasets/minif2f.jsonl", 'r') as file:
    # Each line in the file is a separate JSON object
    data = [json.loads(line.strip()) for line in file.readlines()]

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


comments = None
with open("src/sample-data/comments.txt", 'r') as file:
    comments = [line.strip() for line in file.readlines()]

child = setup_repl()

for problem in tqdm(data):
    informal_prefix = problem['informal_prefix']
    formal_statement = problem['formal_statement']
    PROBLEM_STATEMENT = informal_prefix + formal_statement
    tactic_state = problem['goal']

    game: LeanGame = LeanGame(
        comment_seeds=comments,
        max_depth=20
    )
    state: LeanGameState = game.start_state(
        problem=PROBLEM_STATEMENT,
        tactic_state=tactic_state
    )

    children = []
    while not game.is_terminal(state):
        print(state.human_printout())
        action = 0
        state = game.next_state(state, action)

        input_data = state.pre_LLM_rollout()
        outputs = llm.generate(
            input_data,
            sampling_params=sampling_params
        )
        outputs = outputs[0].outputs[0].text + "\n"
        state.post_LLM_rollout(outputs)
        lean4_input = state.pre_process()
        lean4_output = send_code_read_json({
            "cmd": lean4_input,
            "allTactics": True,
            "tactics": True,
            "env": 0
        }, _child=child)
        state.post_process(lean4_output)
    # save the human printout to a file
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{problem['name']}.txt", 'w') as file:
        file.write(state.human_printout())

    print("Finished problem", problem['name'], "result: ", state.win)
    child.close()
