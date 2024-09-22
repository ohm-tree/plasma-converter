"""
In this file, we will let a human play a game of Lean (using modal).
"""

import cProfile
import io
import os
import pstats
import pexpect
import json
import time
from src.games.lean_game import LeanGame, LeanGameState
from src.networks.prover_llm import ProverLLM
pr = cProfile.Profile()  # Initialize the profiler
pr.enable()              # Start profiling

informal_prefix = r'''/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
'''
formal_statement = r'''theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
    (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
'''
PROBLEM_STATEMENT = informal_prefix + formal_statement
tactic_state = r'''
-- goal:
-- a r : ℝ
-- u : ℕ → ℝ
-- h₀ : ∀ (k : ℕ), u k = a * r ^ k
-- h₁ : u 1 = 2
-- h₂ : u 3 = 6
-- ⊢ u 0 = 2 / √3 ∨ u 0 = -(2 / √3)
'''

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


def send_code_read_json(child: pexpect.spawn, cmd, timeout=20):
    cmd_json = json.dumps(cmd)
    print(cmd_json)
    child.send(cmd_json + "\r\n")
    # Read the input itself.
    # This should be printed instantly, so timeout is set to 1 second.
    child.expect_exact(cmd_json + "\r\n", timeout=1)
    assert child.after.decode('utf-8') == cmd_json + "\r\n"
    print("Sent code to Lean4 REPL.")
    res = ""
    # while res is not a valid json string, read lines.
    # All of the lines should print essentially instantly,
    # so there are no timeouts in this loop.
    start_time = time.time()
    while True:
        res = res + child.readline().decode('utf-8')
        try:
            json.loads(res.strip())
            break
        except json.JSONDecodeError as e:
            print(e)
            pass
        print("---")
        print(res.strip())
        print("---")
        if time.time() - start_time > timeout:
            raise TimeoutError("Lean4 REPL timed out.")
    return json.loads(res)


HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def init_repl(repl):
    send_code_read_json(
        repl,
        {
            "cmd": LEAN4_DEFAULT_HEADER,
            "allTactics": True,
            "tactics": True
        }
    )

repl = pexpect.spawn(
    f"{DEFAULT_LAKE_PATH} exe repl",
    cwd=DEFAULT_LEAN_WORKSPACE)

init_repl(repl)

comments = None
with open("src/sample-data/comments.txt", 'r') as file:
    comments = [line.strip() for line in file.readlines()]

game: LeanGame = LeanGame(
    comment_seeds=comments,
)
state: LeanGameState = game.start_state(
    problem = PROBLEM_STATEMENT,
    tactic_state = tactic_state
)
while not game.is_terminal(state):
    print("Player Must Act".center(80, "#"))
    print(state.human_printout())
    # print each possible comment with the index in front.
    for i, comment in enumerate(comments):
        print(f"{i}: {comment}")
    action = int(input("Enter your action: "))
    # action = comments[action]
    # action = input("Enter your action: ")
    state = game.next_state(state, action)

    print("LLM Must Act".center(80, "#"))
    print(state.human_printout())
    print("Enter your code, line-by-line. Type ``` to quit: \n")
    new_code = ""
    while True:
        line = input()
        if line == "```":
            break
        new_code += line + "\n"
    state.post_LLM_rollout(new_code)

    print("Lean Verifier Must Act!".center(80, "#"))
    print(state.human_printout())
    print("Lean4 Input".center(80, "-"))
    lean4_input = state.pre_process()

    lean4_output = send_code_read_json(repl,
                                    {
                                    "cmd": lean4_input,
                                    "allTactics": True,
                                    "tactics": True,
                                    "env": 0
                                    }, timeout=20)

    lean4_output = send_code_read_json(lean4_input)
    print("Lean4 Output".center(80, "-"))
    print(lean4_output)
    state.post_process(lean4_output)


print("Terminated!")
state.process()
print(state.human_printout())

pr.disable()             # Stop profiling

# Create an output stream to capture the profiling stats
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
# save stats to a file
with open("profile_stats.txt", "w") as f:
    ps = pstats.Stats(pr, stream=f).sort_stats(sortby)
    ps.print_stats()

# Print or log the profiling results
print(s.getvalue())
