"""
In this file, we will let a human play a game of Lean (using modal).
"""

import cProfile
import io
import json
import os
import pstats
import time

import pexpect
from vllm import LLM, SamplingParams

from src.games.lean_game import Lean, LeanState

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def load_problem_minif2f(problem_name="algebra_bleqa_apbon2msqrtableqambsqon8b"):
    data = None
    with open(f"{HOME_DIR}/plasma-converter/datasets/minif2f.jsonl", 'r') as file:
        # Each line in the file is a separate JSON object
        data = [json.loads(line.strip()) for line in file.readlines()]
    for problem in data:
        if problem['name'] == problem_name:
            return problem


problem = load_problem_minif2f()

informal_prefix = problem['informal_prefix']
formal_statement = problem['formal_statement']
PROBLEM_STATEMENT = informal_prefix + formal_statement

tactic_state = problem['goal']


llm = LLM(model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
          max_num_batched_tokens=8192,
          trust_remote_code=True,
          dtype="float16",
          tensor_parallel_size=4)

sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=0.0,
    top_k=1,
    top_p=1.0
)


# useful copy+paste for the game.
"""
  -- First, we want to re-write the condition about the second
  -- and fourth terms of the geometric sequence using the definition of a geometric sequence
  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero,
    Nat.succ_add]
  have h₁' : a * r = 2 := by simpa [h₀] using h₁
  have h₂' : a * r ^ 3 = 6 := by simpa [h₀] using h₂
  -- Now we can divide the two equations to eliminate $a$ and determine $r$
  have h₃ : r ^ 2 = 3 := by
    nlinarith
  -- Finally, we can substitute back to find $a$
  have h₄ : a = 2 / Real.sqrt 3 ∨ a = -(2 / Real.sqrt 3) := by
    apply eq_or_eq_neg_of_sq_eq_sq <;>
    field_simp <;>
    nlinarith
  simpa [h₀] using h₄
"""


def send_code_read_json(cmd, timeout_start=30, timeout_finish=30):

    child = pexpect.spawn(
        f"{DEFAULT_LAKE_PATH} exe repl",
        cwd=DEFAULT_LEAN_WORKSPACE)

    cmd_json = json.dumps(cmd)
    print(cmd_json)
    child.send(cmd_json + "\r\n")
    # Read the input itself.
    # This should be printed instantly, so timeout is set to 1 second.
    child.expect_exact(cmd_json + "\r\n", timeout=20)
    assert child.after.decode('utf-8') == cmd_json + "\r\n"
    print("Sent code to Lean4 REPL.")

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
    child.close()
    return json.loads(res)


comments = None
with open("src/sample-data/comments.txt", 'r') as file:
    comments = [line.strip() for line in file.readlines()]

game: Lean = Lean(
    comment_seeds=comments,
)
state: LeanState = game.start_state(
    problem=PROBLEM_STATEMENT,
    tactic_state=tactic_state
)
while not game.is_terminal(state):
    print("Player Must Act".center(80, "#"))
    print(state.human_printout())

    # print each possible comment with the index in front.
    for i, comment in enumerate(comments):
        print(f"{i}: {comment}")
    action = int(input("Enter your action (int): "))
    # action = comments[action]
    # action = input("Enter your action: ")
    state = game.next_state(state, action)

    print("LLM Must Act".center(80, "#"))
    print(state.human_printout())
    input_data = state.pre_LLM_rollout()
    print(input_data)
    outputs = llm.generate(
        input_data,
        sampling_params=sampling_params
    )
    outputs = outputs[0].outputs[0].text

    print(outputs)

    state.post_LLM_rollout(outputs)

    print("Lean Verifier Must Act!".center(80, "#"))
    print(state.human_printout())
    print("Lean4 Input".center(80, "-"))
    lean4_input = state.pre_process()

    lean4_output = send_code_read_json({
        "cmd": lean4_input,
        "allTactics": True,
        "tactics": True,
        "ast": True,
    })

    print("Lean4 Output".center(80, "-"))
    print(str(lean4_output)[:100])
    print(f"({len(str(lean4_output))} characters)")
    state.post_process(lean4_output)


print("Terminated!")
print(state.human_printout())
