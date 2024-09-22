"""
In this file, we will let a human play a game of Lean (using modal).
"""

import cProfile
import io
import os
import pstats

from src.games.lean_game import LeanGame, LeanGameState
from src.networks.prover_llm import ProverLLM

pr = cProfile.Profile()  # Initialize the profiler
pr.enable()              # Start profiling

PROBLEM_STATEMENT = r'''/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
    (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
'''

comments = None
with open("src/sample-data/comments.txt", 'r') as file:
    comments = [line.strip() for line in file.readlines()]

game: LeanGame = LeanGame(
    comment_seeds=comments,
)
state: LeanGameState = game.start_state(PROBLEM_STATEMENT)
while not game.is_terminal(state):
    print("HUMAN", state.human_printout())
    # print each possible comment with the index in front.
    for i, comment in enumerate(comments):
        print(f"{i}: {comment}")
    action = int(input("Enter your action: "))
    # action = comments[action]
    # action = input("Enter your action: ")
    state = game.next_state(state, action)
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
