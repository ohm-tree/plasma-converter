"""
In this file, we will let a human play a game of Lean (using modal).
"""

import os

import modal

app = modal.App("lean-game-test")

image = modal.Image.from_registry("czhang2718/deepseek-lean-ubuntu-py")
image = image.pip_install("pydantic", extra_options="-U")


# currently, __file__ is /src/tests/LLM_speed_test.py
# we want to get to the local root folder.
local_root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))


# run with `modal run -i lean_game_test.py`
@app.function(image=image,
              gpu=modal.gpu.A100(count=1, size="80GB"),
              timeout=86400,
              mounts=[modal.Mount.from_local_dir(
                  os.path.join(local_root_dir, "src"), remote_path="/root/src"),
                  modal.Mount.from_local_dir(
                  os.path.join(local_root_dir, "prover"), remote_path="/root/prover")]
              )
def human_play_lean_game() -> None:
    import cProfile
    import io
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
        completion_model=ProverLLM(),
    )
    state: LeanGameState = game.start_state(PROBLEM_STATEMENT)
    modal.interact()
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


@app.local_entrypoint()
def main():
    human_play_lean_game.remote()
