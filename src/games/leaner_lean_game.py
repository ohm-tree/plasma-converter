import os
import time
import traceback
from pprint import pprint
from typing import Callable, List, Optional

import numpy as np

from src.games.game import Game

# from src.networks.prover_llm import ProverLLM

############################################# Spaghetti Code from DeepSeek (TODO: Rewrite) #############################################

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


class AttrDict(dict):
    """A dictionary that allows attribute-style access."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


############################################# Lean Game Formalism #############################################


class LeanGameStateError(Exception):
    pass


class LeanGameState:
    def __init__(self,
                 problem: str,
                 old_code: str,
                 old_tactic_state: str,
                 comment: str,
                 depth: int,
                 header: str = LEAN4_DEFAULT_HEADER,
                 rollout_done: bool = False,
                 processed: bool = False,
                 new_code: Optional[str] = None,
                 win: Optional[bool] = None,
                 dead: Optional[bool] = None,
                 tactic_state: Optional[str] = None,
                 valid_code: Optional[str] = None,
                 ):
        """
        A LeanGameState can be in one of three states:
        1. Just-initialized. In this case, neither the LLM
        rollout nor the Lean verification has been run.
        2. Post-rollout. In this case, the LLM rollout has
        been run, but the Lean verification has not been run.
        3. Fully processed. In this case, both the LLM
        rollout and the Lean verification have been run.

        Here are the fields and their statuses in each state:

        | Field            | Just-initialized | Post-rollout | Fully processed |
        | ---------------- | ---------------- | ------------ | --------------- |
        | problem          | Required         | Required     | Required        |
        | old_code         | Required         | Required     | Required        |
        | old_tactic_state | Required         | Required     | Required        |
        | comment          | Required         | Required     | Required        |
        | depth            | Required         | Required     | Required        |
        | header           | Required         | Required     | Required        |
        | rollout_done     | False            | True         | True            |
        | processed        | False            | False        | True            |
        | new_code         | None             | Required     | Required        |
        | win              | None             | None         | Required        |
        | dead             | None             | None         | Required        |
        | tactic_state     | None             | None         | Required        |
        | valid_code       | None             | None         | Required        |

        Attempting to call methods like LeanGame.is_terminal(),
        Leangame.reward(), or LeanGame.next_state() on a
        non-fully-processed state will raise a LeanGameStateError.

        Parameters
        ----------
        problem: str
            The problem statement to be solved.
        old_code: str
            The code that has been written so far.
            The old code always includes comments.
        old_tactic_state: str
            The tactic state after the old code was added;
            used for the LLM prompt.
        comment: str
            The comment that was added to the code; this is the action taken.
        depth: int
            The depth of the proof.
        header: str
            The header for the Lean code.
        rollout_done: bool
            Whether the LLM rollout has been done.
        processed: bool
            Whether the Lean verification has been done.
        new_code: Optional[str]
            The new code that was added to the proof.
            This will generally not be truncated to a valid proof
            nor truncated to the first goal change.
            This will be None if rollout_done is False.
            This will not contain the comment.
        win: Optional[bool]
            Whether the proof is complete. This will be None
            if the proof is not fully processed.
        dead: Optional[bool]
            Whether the proof is dead. This will be None
            if the proof is not fully processed.
        tactic_state: Optional[str]
            The tactic state after the new code was added.
            This will be None if the proof is not fully processed.
        valid_code: Optional[str]
            The new code that was added to the proof.
            This will be None if the proof is not fully processed.
        """
        if rollout_done:
            if processed:
                if new_code is None or win is None or dead is None or tactic_state is None or valid_code is None:
                    raise ValueError(
                        "If rollout_done is True and processed is True, new_code, win, tactic_state, and valid_code must all be non-None.")
            else:
                if new_code is None:
                    raise ValueError(
                        "If rollout_done is True and processed is False, new_code must be non-None.")
                if win is not None or dead is not None or tactic_state is not None or valid_code is not None:
                    raise ValueError(
                        "If rollout_done is True and processed is False, win, tactic_state, and valid_code must all be None.")
        else:
            if processed:
                raise ValueError(
                    "If rollout_done is False, processed must be False.")
            if new_code is not None or win is not None or dead is not None or tactic_state is not None or valid_code is not None:
                raise ValueError(
                    "If rollout_done is False, new_code, win, tactic_state, and valid_code must all be None.")

        self.problem: str = problem
        self.old_code: str = old_code
        self.old_tactic_state: str = old_tactic_state
        self.comment: str = comment
        self.depth: int = depth
        self.header: str = header
        self.rollout_done: bool = rollout_done
        self.processed: bool = processed
        self.new_code: Optional[str] = new_code
        self.win: Optional[bool] = win
        self.dead: Optional[bool] = dead
        self.tactic_state: Optional[str] = tactic_state
        self.valid_code: Optional[str] = valid_code

        # We don't want to re-generate a child when we re-do an action,
        # so pointers to the children are stored here.
        self.children = {}

    def __hash__(self) -> int:
        return hash(
            (self.problem, self.old_code, self.old_tactic_state, self.comment, self.depth, self.header,
             self.rollout_done, self.processed, self.new_code, self.win, self.dead, self.tactic_state, self.valid_code)
        )

    def add_child(self, action: int, child: 'LeanGameState'):
        if not self.processed:
            raise LeanGameStateError(
                "Cannot add a child to a LeanGameState if it has not been processed.")
        self.children[action] = child

    def terminal(self) -> bool:
        if not self.processed:
            raise LeanGameStateError(
                "Cannot check if a LeanGameState is terminal if it has not been processed.")
        return self.win or self.dead

    def human_printout(self) -> str:
        res = ""

        def fancy_field(name: str, value: str, length=80, tick='-') -> str:
            res = name.center(length, tick) + "\n"
            res += value
            if len(value) == 0:
                res += "[Empty Field]\n"
            elif value[-1] != '\n':
                res += '\n'
                res += "[Missing newline]\n"
            return res

        if self.rollout_done:
            if self.processed:
                status_code = "Fully processed\n"
            else:
                status_code = "Rollout done\n"
        else:
            status_code = "Just initialized\n"

        res += fancy_field("Status", status_code)
        res += fancy_field("Header", self.header)
        res += fancy_field("Problem", self.problem)
        res += fancy_field("Old Code", self.old_code)
        res += fancy_field("Comment", self.comment)
        if self.processed:
            res += fancy_field("Valid Truncation of New Code", self.valid_code)
        elif self.rollout_done:
            res += fancy_field("Completed Rollout without Truncation",
                               self.new_code)
        else:
            res += fancy_field("[New Code will be here]", "\n")

        res += fancy_field("Old Tactic State", self.old_tactic_state)
        if self.processed:
            res += fancy_field("New Tactic State", self.tactic_state)

        res += fancy_field("Meta", f"Processed: {self.processed}, Rollout Done: {self.rollout_done}\n"
                           f"Win: {self.win}, Dead: {self.dead}\n"
                           f"Depth: {self.depth} Number of Children: {len(self.children)}\n")
        return res

    def human_json(self) -> dict:
        """
        Returns a JSON representation of the game state.
        This is meant to be stored alongside game data
        by the workers so that we can debug easier.
        """
        return {
            "problem": self.problem,
            "old_code": self.old_code,
            "old_tactic_state": self.old_tactic_state,
            "comment": self.comment,
            "depth": self.depth,
            "header": self.header,
            "rollout_done": self.rollout_done,
            "processed": self.processed,
            "new_code": self.new_code,
            "win": self.win,
            "tactic_state": self.tactic_state,
            "valid_code": self.valid_code,
        }

    def __str__(self) -> str:
        return f"LeanGameState({self.problem}, {self.code}, processed = {self.processed})"

    def pre_LLM_rollout(self) -> str:
        """
        This function is called before the LLM rollout is done.
        It generates a prompt for the LLM.
        """
        if self.rollout_done:
            raise LeanGameStateError(
                "Should not LLM-pre-process a LeanGameState that has already had an LLM rollout.")

        return 'Complete the following Lean 4 code.\nThe tactic state is:\n' + \
            self.old_tactic_state+'\n```lean\n' + self.header + self.problem + \
            self.old_code + self.comment

    def post_LLM_rollout(self, new_code: str):
        """
        This function is called after the LLM rollout is done.
        """
        if self.rollout_done:
            raise LeanGameStateError(
                "Should not LLM-post-process a LeanGameState that has already had an LLM rollout.")
        if new_code.endswith('```'):
            new_code = new_code[:-3]
        self.new_code = new_code
        self.rollout_done = True

    def pre_policy_value(self) -> str:
        """
        This function is called before the policy value network is called.
        It generates a prompt for the policy value network.
        """
        if not self.processed:
            raise LeanGameStateError(
                "Should not pre-process a LeanGameState that has not been processed.")

        # TODO: make this better.

        return 'Complete the following Lean 4 code.\n```lean\n' + self.header + self.problem + self.old_code + self.comment + self.new_code

    def pre_process(self) -> str:
        """
        This function is called before the state is processed.
        It prepares a string query for the lean 4 verifier.
        """

        if self.processed:
            raise LeanGameStateError(
                "Should not pre-process a LeanGameState that has already been processed.")

        return ''.join(
            [self.problem, self.old_code,
                self.comment, self.new_code]
        )

    @ classmethod
    def get_index(s: str, row: int, col: int) -> int:
        """
        Convert a (row, col) pair to an index in a string.
        The Lean 4 repl convention is that rows are 1-indexed
        and columns are 0-indexed.
        """
        lines = s.split('\n')
        # Add back the newline for accurate indexing.
        line_lengths = [len(line) + 1 for line in lines]
        if row < 1:
            raise ValueError("Row must be at least 1.")
        if col < 0:
            raise ValueError("Column must be at least 0.")
        if row > len(lines):
            raise ValueError("Row is too large.")
        if col >= line_lengths[row-1]:
            # The col should never be exactly line_lengths;
            # If the cursor is "after the newline character"
            # then it should really be the 0th index of the next line.
            raise ValueError("Column is too large.")
        return sum(line_lengths[:row-1]) + col

    def truncate(self, sorries: List[dict], errors: List[dict]):
        """
        First, we need to find the last valid tactic.

        Unsolved goals also show up as errors, and we ignore them;
        they have been ommitted from the errors list in post_process()
        already.

        Parameters
        ----------
        sorries: List[dict]
            A list of sorry messages.
        errors: List[dict]
            A list of error messages.

        Modifies
        -------
        self.valid_code: str
            The new code that was added to the proof.
            This will be truncated to the first error or sorry.
        """
        code_no_header = ''.join(
            [self.problem, self.old_code,
                self.comment, self.new_code]
        )

        _prefix_len = len(self.problem)
        truncate_pos = len(code_no_header)
        for info in sorries:
            info_pos = self.get_index(
                code_no_header, info['pos']['line'], info['pos']['column'])
            if info_pos >= _prefix_len:
                truncate_pos = min(truncate_pos, info_pos)
        for info in errors:
            info_pos = self.get_index(
                code_no_header, info['pos']['line'], info['pos']['column'])
            if info_pos >= _prefix_len:
                truncate_pos = min(truncate_pos, info_pos)

        self.valid_code = code_no_header[:truncate_pos]
        assert self.valid_code.startswith(self.problem)
        self.valid_code = self.valid_code[len(self.problem):]
        assert self.valid_code.startswith(self.old_code)
        self.valid_code = self.valid_code[len(self.old_code):]
        assert self.valid_code.startswith(self.comment)
        self.valid_code = self.valid_code[len(self.comment):]

    def get_goals(self, goals: List[dict]):
        """
        Get the the most recent goal from the Lean 4 verification.

        Parameters
        ----------
        goals: List[dict]
            A list of goal messages.

        Modifies
        -------
        self.tactic_state: str
            The tactic state after the new code was added.
        """

        self.tactic_state = ""
        max_line = float('-inf')
        max_column = float('-inf')
        for tactic in goals:
            if (tactic['pos']['line'] < max_line) or (tactic['pos']['line'] == max_line and tactic['pos']['column'] < max_column):
                max_line = tactic['pos']['line']
                max_column = tactic['pos']['column']
                self.tactic_state = tactic['data'].lstrip()[
                    len("unsolved goals\n"):]

    def post_process(self, repl_result: dict):
        """
        This function is called after the state is processed.

        We compute the following things:
            1. We check if the result is 'complete'. In that case,
            we win, and we can break immediately.
            2. We will first truncate everthing after the first error
            occurs. Then, we will truncate everything after the first
            *new* tactic state. Ideally, this means that our proof has
            grown by exactly one new tactic state.
            This is called "segmentation".
            If at this stage, *no new code is written* (possibly
            because the very next chunk of code causes an error,
            then we need to set self.dead = True).
            3. The new tactic state after this new segment.
            This will be stored in self.new_tactic_state.
            4. Whether or not we are done. This will
            be stored in self.win.

        Parameters
        ----------
        repl_result: dict
            The result of the Lean 4 verification.

        """
        self.processed = True

        if repl_result.get('system_error', False):
            self.tactic_state = ""
            self.valid_code = ""
            self.dead = True
            self.win = False
            return

        sorries = repl_result.get('sorries', [])
        tactics = repl_result.get('tactics', [])

        errors = []
        warnings = []
        infos = []
        goals = []

        complete = not sorries
        for m in repl_result.get('messages', []):
            if m['severity'] == 'error':
                complete = False
                if m['data'].lstrip().startswith("unsolved goals\n"):
                    goals.append(m)
                else:
                    errors.append(m)
            elif m['severity'] == 'warning':
                if "declaration uses 'sorry'" in m['data'] or 'failed' in m['data']:
                    complete = False
                warnings.append(m)
            elif m['severity'] == 'info':
                infos.append(m)

        self.truncate(sorries, errors)
        self.get_goals(goals)
        if self.new_code.strip() == "":
            # This means that the new code was truncated to nothing,
            # i.e. the first new line of code written caused an error.
            # This is a dead state.
            self.dead = True
            self.win = False
            return

        if complete:
            self.dead = False
            self.win = True
            return

        self.dead = False
        self.win = False

    def code(self) -> str:
        """
        Returns the full code of the state.
        old_code + comment + new_code

        Returns
        -------
        str
            The full code of the state.
        """
        if not self.processed:
            raise LeanGameStateError(
                "Cannot get the code of a LeanGameState that has not been processed.")
        return ''.join(
            [self.old_code, self.comment, self.valid_code]
        )

    @ classmethod
    def saves(cls, states: List['LeanGameState'], filename: str):
        """
        Save a collection of states to a file.

        This is super important. During MCTS,
        this is what gets saved to the replay buffer.
        """
        np.save(filename, [state.human_json() for state in states])

    @ classmethod
    def loads(cls, filename: str) -> List['LeanGameState']:
        """
        Load a collection of states from a file.
        """
        states = np.load(filename, allow_pickle=True)
        return [cls(**state) for state in states]


class LeanGame(Game[LeanGameState]):
    """
    The LeanGame class implements the game logic
    for a Lean 4 proof assistant game.
    """

    def __init__(self,
                 comment_seeds: List[str],
                 max_depth: Optional[int] = 50,
                 max_completion_len: int = 4000):
        """
        In this game, ACTIONs are comments that can be added to the proof.
        Each action is followed by an LLM response, which appends to the
        proof state.

        There are three terminal conditions:
        - The proof is complete!
        - The proof is dead (invalid).
        - The maximum depth of the proof is reached.

        Parameters
        ----------
        comment_seeds: List[str]
            The list of comments that can be used as actions.
        max_depth: Optional[int]
            The maximum depth of the proof.
        max_completion_len: int
            The maximum length of the completion.
        """
        self.comment_seeds = comment_seeds or []
        self.num_comment_seeds = len(comment_seeds)
        self.max_depth = max_depth

        self.max_completion_len = max_completion_len

    def start_state(self,
                    problem: str,
                    header: str = LEAN4_DEFAULT_HEADER,
                    tactic_state: str = ""
                    ) -> LeanGameState:
        """
        Returns the initial state of the game.

        Parameters
        ----------
        problem: str
            The problem statement to be solved.
        header: str
            The header for the Lean code.
        tactic_state: str
            The initial tactic state.
        """

        return LeanGameState(
            problem=problem,
            old_code="",
            old_tactic_state="",
            comment="",
            depth=0,
            header=header,
            rollout_done=True,
            processed=True,
            new_code="",
            win=False,
            dead=False,
            tactic_state=tactic_state,
            valid_code=""
        )

    def next_state(self, state: LeanGameState, action: int) -> LeanGameState:
        """
        Returns the next state of the game given a current state and action.
        Requires that the state is non-terminal and action is legal.
        """

        if action in state.children:
            # We've already computed this child, and we can return the cached result.
            return state.children[action]

        if not state.processed:
            raise LeanGameStateError(
                "Cannot get the next state of a LeanGameState that has not been processed.")

        if state.terminal():
            raise LeanGameStateError(
                "Cannot get the next state of a terminal LeanGameState.")

        comment = self.comment_seeds[action]

        new_state = LeanGameState(
            problem=state.problem,
            old_code=state.code(),
            old_tactic_state=state.tactic_state,
            comment=comment,
            depth=state.depth + 1,
            header=state.header,
            rollout_done=False,
            processed=False,
        )

        state.add_child(action, new_state)
        return new_state

    def action_mask(self, state: LeanGameState) -> np.ndarray:
        """
        # TODO: In this game, the action mask is redundant because
        all actions are always allowed.
        Drop this at some point. Maybe we won't drop it because we want the MCTS
        code to be clean and generalizable. We'll see, it's epsilon right now.
        """
        return np.ones(self.num_comment_seeds, dtype=bool)

    def is_terminal(self, state: LeanGameState) -> bool:
        """
        The game is considered over if the board is fully filled or if the state is dead.
        """
        return state.terminal() or state.depth > self.max_depth

    def reward(self, state: LeanGameState) -> float:
        """
        Rewards the player if the board is correctly completed, otherwise 0.
        """
        assert self.is_terminal(
            state), "Reward can only be calculated for terminal states."
        return 1.0 if state.win else -1.0

    def display_state(self, state: LeanGameState) -> str:
        """
        Returns a string representation of the game state, marking dead states.
        """
        return str(state)

    def hash_state(self, state: LeanGameState) -> int:
        """
        Returns a hash of the game state.
        """
        return hash(state.code)
