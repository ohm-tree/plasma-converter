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

from src.games.lean_game import LeanGame, LeanGameState

# set "VLLM_LOGGING_LEVEL" to "WARNING" to suppress logging
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

HOME_DIR = os.path.expanduser('~')
print("HOME_DIR", HOME_DIR)

with open(f"{HOME_DIR}/plasma-converter/datasets/minif2f.jsonl", 'r') as file:
    # Each line in the file is a separate JSON object
    data = [json.loads(line.strip()) for line in file.readlines()]


llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
          max_num_batched_tokens=8192,
          trust_remote_code=True,
          dtype="float16",
          tensor_parallel_size=4)

# add a custom stopping token
# which stops on newlines
sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    stop=["\n"]
)


def send_code_read_json(cmd, timeout_start=30, timeout_finish=30, _child: Optional[pexpect.spawn] = None, kill=False):
    try:
        return _send_code_read_json(cmd, timeout_start=timeout_start, timeout_finish=timeout_finish, _child=_child, kill=kill)
    except Exception as e:
        print(e)
        return {'system_error': str(e)}


def _send_code_read_json(cmd, timeout_start=30, timeout_finish=30, _child: Optional[pexpect.spawn] = None, kill=False):
    if _child is None:
        child = pexpect.spawn(
            f"{DEFAULT_LAKE_PATH} exe repl",
            cwd=DEFAULT_LEAN_WORKSPACE)
    else:
        child = _child

    cmd_json = json.dumps(cmd)
    # print(cmd_json)
    child.send(cmd_json + "\r\n")
    # Read the input itself.
    # This should be printed instantly, so timeout is set to 1 second.
    child.expect_exact(cmd_json + "\r\n", timeout=20)
    assert child.after.decode('utf-8') == cmd_json + "\r\n"
    # print("Sent code to Lean4 REPL.")

    # Read the output.
    # This code is critical; the repl seems to print out some
    # strange non-json stuff before the actual json output,
    # including characters that delete the previous input,
    # such that it doesn't show up in debug output.
    child.expect_exact("{", timeout=timeout_start)
    res = "{"
    print("Received start of output from Lean4 REPL.")
    # while res is not a valid json string, read lines.
    # All of the lines should print essentially instantly,
    # so there are no timeouts in this loop.
    start_time = time.time()
    while True:
        res = res + child.readline().decode('utf-8')
        try:
            # print all chars in res
            json.loads(res.strip())
            break
        except json.JSONDecodeError as e:
            # print(e)
            pass
        if time.time() - start_time > timeout_finish:
            raise TimeoutError("Lean4 REPL timed out.")
        # time.sleep(0.1)

    # kill
    if kill:
        child.close()
    return json.loads(res)


def setup_repl():
    child = pexpect.spawn(
        f"{DEFAULT_LAKE_PATH} exe repl",
        cwd=DEFAULT_LEAN_WORKSPACE)
    send_code_read_json(
        {
            "cmd": LEAN4_DEFAULT_HEADER,
            "allTactics": True,
            "tactics": True,
        },
        _child=child
    )
    return child


HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


comments = None
with open("src/sample-data/comments.txt", 'r') as file:
    comments = [line.strip() for line in file.readlines()]


for problem in tqdm(data):
    child = setup_repl()
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
    with open(f"outputs/{problem['name']}.txt", 'w', encoding='utf-8') as file:
        file.write(state.human_printout())

    print("Finished problem", problem['name'], "result: ", state.win)
    child.close()
