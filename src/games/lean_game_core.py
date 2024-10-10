import os
import random
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple

import numpy as np

from src.games.concurrent import ConcurrentClass, handler, on_startup
from src.games.game import ConcurrentGameState
from src.uct.uct_node import UCTNode
from src.workers.types import LeanTaskType
from src.workers.worker import (
    TaskIdentifier,
    TaskType,
    WorkerIdentifer,
    WorkerResponse,
    WorkerTask,
)

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\nset_option linter.all false\nopen BigOperators Real Nat Topology Rat\n\n"


class LeanGameStateError(Exception):
    pass


@dataclass(frozen=True)  # Hashable
class LeanGameMove:
    new_code: str


class LeanGameState(ConcurrentGameState[LeanGameMove]):
    """
    The LeanGameState class implements the game state for a Lean 4 proof assistant game.
    This is the "core" of the game, and does not involve implementation details which
    should be delegated to a higher-level algorithm for playing the game, such as comment
    generation etc etc.

    A move simply appends text to the end of the code. Afterwards, the text is truncated
    to the location of the first error. If this truncation causes the code to be the same
    length (or less) than it was before, then the game terminates in a loss with a reward
    of -1. We call this a dead state. Post-truncation, the code may also be correct. In
    this case, the code terminates in a win with a reward of 1. If the code is neither dead
    nor correct, then the game continues.
    """

    # global UUID
    UUID = 0

    def __init__(self,
                 parent: Optional['LeanGameState'],
                 problem: str,
                 header: str,
                 new_code: Optional[str],
                 depth: int,
                 max_depth: int,
                 ready: bool = False,
                 win: Optional[bool] = None,
                 dead: Optional[bool] = None,
                 tactic_state: Optional[str] = None,
                 valid_code: Optional[str] = None,
                 **kwargs
                 ):
        """
        Here are the fields and their statuses in each state:

        | Field            | Initialized     | Ready    |
        | ---------------- | --------------- | -------- |
        | parent           | Required        | Required |
        | problem          | Required        | Required |
        | header           | Required        | Required |
        | new_code         | Not Required    | Required |
        | ready            | False           | True     |
        | win              | None            | Required |
        | dead             | None            | Required |
        | tactic_state     | None            | Required |
        | valid_code       | None            | Required |

        Parameters
        ----------
        parent: Optional[LeanGameState]
            The parent state. If None, this is the root state.
        problem: str
            The problem statement to be solved.
        header: str
            The header for the Lean code.
        new_code: str
            The new code that was added to the proof.
            This will generally not be truncated to a valid proof
            nor truncated to the first goal change.
        ready: bool
            Whether the proof is ready. This will be False
            if the proof is not fully processed.
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
        super().__init__(**kwargs)
        self.parent: Optional[LeanGameState] = parent
        self.problem: str = problem
        self.header: str = header
        self.new_code: str = new_code
        self.depth: int = depth
        self.max_depth: int = max_depth

        self.old_tactic_state = self.parent.tactic_state if self.parent else ""
        self.old_code = self.parent.new_code if self.parent else ""

        self._win: Optional[bool] = win
        self._dead: Optional[bool] = dead
        self.tactic_state: Optional[str] = tactic_state
        self.valid_code: Optional[str] = valid_code

        self._ready = ready

        self.check_state()

        # We don't want to re-generate a child when we re-do an action,
        # so pointers to the children are stored here.
        self.children = {}

        # This is just a unique identifier for the state.
        self._id = LeanGameState.UUID
        LeanGameState.UUID += 1

    ################################################################
    # Utilities
    ################################################################

    ################################################################
    # Implementing the ConcurrentGameState Interface
    ################################################################

    @classmethod
    def saves(cls, states: List['LeanGameState'], filename: str):
        """
        Save a collection of states to a file.

        This is super important. During MCTS,
        this is what gets saved to the replay buffer.
        """
        np.save(filename, [state.human_json() for state in states])

    @classmethod
    def loads(cls, filename: str) -> List['LeanGameState']:
        """
        Load a collection of states from a file.
        """
        states = np.load(filename, allow_pickle=True)
        return [cls(**state) for state in states]

    def human_json(self):
        """
        Return a human-readable JSON representation of the state.
        """

        return {
            "header": self.header,
            "problem": self.problem,
            "old_code": self.old_code,
            "new_code": self.new_code,
            "valid_code": self.valid_code,
            "tactic_state": self.tactic_state,
            "old_tactic_state": self.old_tactic_state,
            "win": self._win,
            "dead": self._dead,
            "depth": self.depth,
        }

    @classmethod
    def starting_state(cls,
                       worker_id: WorkerIdentifer,
                       problem: str,
                       header: str,
                       tactic_state: str,
                       max_depth: int = 100) -> 'LeanGameState':
        """
        Returns the starting state of the game.
        """
        return LeanGameState(
            parent=None,
            problem=problem,
            header=header,
            new_code="",
            depth=0,
            max_depth=max_depth,
            ready=True,
            win=False,
            dead=False,
            tactic_state=tactic_state,
            valid_code="",
            worker_id=worker_id
        )

    def check_state(self) -> None:
        """
        Check that the state is valid.
        """
        if None in [self.problem, self.header]:
            raise ValueError(
                "problem, header are required.")

        if self.ready():
            if None in [self._win, self._dead, self.tactic_state, self.valid_code]:
                raise ValueError(
                    "win, dead, tactic_state, and valid_code are required if ready is True.")
        else:
            if [self._win, self._dead, self.tactic_state, self.valid_code].count(None) != 4:
                raise ValueError(
                    "win, dead, tactic_state, and valid_code must be None if ready is False.")

    def next_state(self, action: LeanGameMove) -> 'LeanGameState':
        """
        Returns the next state of the game given a current state and action.
        Requires that the state is non-terminal and action is legal.

        This method cannot be called if the state is terminal or not ready!
        Check with terminal() and ready() before calling this method.

        Parameters
        ----------
        action: LeanGameMove
            The action to be taken.
        """
        if not self.ready():
            raise LeanGameStateError(
                "Cannot get the next state of a LeanGameState that has not been processed.")

        if self.terminal():
            raise LeanGameStateError(
                "Cannot get the next state of a terminal LeanGameState.")

        if action in self.children:
            # We've already computed this child, and we can return the cached result.
            return self.children[action]

        new_state = LeanGameState(
            parent=self,
            problem=self.problem,
            header=self.header,
            new_code=self.new_code,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            ready=False,
            win=None,
            dead=None,
            tactic_state=None,
            valid_code=None,
            worker_id=self.worker_id
        )
        self.children[action] = new_state
        return new_state

    def terminal(self) -> bool:
        """
        We should be able to check if we're terminal
        even if we haven't done the PV or comment generation
        yet; this is useful because I have a lot of while not state.terminal()
        loops in the code.
        """
        if not self.ready():
            raise LeanGameStateError(
                "Cannot check if a LeanGameState is terminal if it has not been processed.")
        return self._win or self._dead or self.depth >= self.max_depth

    def reward(self) -> float:
        """
        Returns a float consisting of the reward for the player.
        """
        if not self.ready():
            raise LeanGameStateError(
                "Cannot get the reward of a LeanGameState that has not been processed.")
        if not self.terminal():
            raise LeanGameStateError(
                "Cannot get the reward of a non-terminal LeanGameState.")
        return 1.0 if self._win else -1.0

    def __hash__(self) -> int:
        """
        Hash the state based on the id.
        """
        return self._id

    def make_root(self) -> None:
        """
        Severs any references to the parent of
        this state, making it the root of the tree.
        """
        if self.parent is not None:
            self.parent.children = {}
            self.parent = None

    ################################################################
    # Details of the ConcurrentClass Interface
    ################################################################

    @on_startup
    def pre_process(self) -> Iterator[WorkerTask]:
        """
        This function is called before the state is processed.
        It prepares a string query for the lean 4 verifier.
        """

        if self.ready():
            raise LeanGameStateError(
                "Should not pre-process a LeanGameState that has already been processed.")

        yield WorkerTask(
            head_id=self.worker_id,
            task_id=TaskIdentifier(task_type=LeanTaskType, task_idx=self._id),
            task=self.problem + self.old_code + self.new_code
        )

        self.finish()

    @classmethod
    def get_index(cls, s: str, row: int, col: int) -> int:
        """
        Convert a (row, col) pair to an index in a string.
        The Lean 4 repl convention is that rows are 1-indexed
        and columns are 0-indexed.
        """
        lines = s.split('\n')
        # Add back the newline for accurate indexing.
        line_lengths = [len(line) + 1 for line in lines]
        if row < 1:
            raise ValueError(
                f"Row must be at least 1. row = {row}, col = {col}, line_lengths = {line_lengths}, s = {s}")
        if col < 0:
            raise ValueError(
                f"Column must be at least 0. row = {row}, col = {col}, line_lengths = {line_lengths}, s = {s}")
        if row > len(lines):
            raise ValueError(
                f"Row is too large. row = {row}, col = {col}, line_lengths = {line_lengths}, s = {s}")
        if col >= line_lengths[row-1]:
            # The col should never be exactly line_lengths;
            # If the cursor is "after the newline character"
            # then it should really be the 0th index of the next line.
            raise ValueError(
                f"Column is too large. row = {row}, col = {col}, line_lengths = {line_lengths}, s = {s}")
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
                self.new_code]
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
        # I guess this might not be true in some edge cases?
        # assert self.valid_code.startswith(self.old_code)
        self.valid_code = self.valid_code[len(self.old_code):]

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

    @handler(LeanTaskType)
    def post_process(self, result: WorkerResponse):
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
            then we need to set self._dead = True).
            3. The new tactic state after this new segment.
            This will be stored in self.new_tactic_state.
            4. Whether or not we are done. This will
            be stored in self._win.

        Parameters
        ----------
        repl_result: dict
            The result of the Lean 4 verification.

        """
        if self.ready():
            raise LeanGameStateError(
                "Should not post-process a LeanGameState that has already been processed.")

        repl_result: dict = result.response

        if 'system_error' in repl_result or 'message' in repl_result or ('ast' not in repl_result):
            self.tactic_state = ""
            self.valid_code = ""
            self._dead = True
            self._win = False
            return

        # 'sorries' has never been part of a repl_result
        # if 'sorries' in repl_result:
            # raise ValueError(
            # "Unexpected key 'sorries' in repl_result.")

            # tactics = repl_result.get('tactics', [])

        errors = []
        warnings = []
        infos = []
        goals = []
        sorries = repl_result.get('sorries', [])

        if len(sorries) > 0:
            complete = False

        complete = False
        for m in repl_result.get('messages', []):
            if m['severity'] == 'error':
                complete = False
                if m['data'].lstrip().startswith("unsolved goals\n"):
                    goals.append(m)
                else:
                    errors.append(m)
            elif m['severity'] == 'warning':
                if "declaration uses 'sorry'" in m['data']:
                    sorries.append(m)
                    complete = False
                if 'failed' in m['data']:
                    complete = False

                warnings.append(m)
                # There exist some warnings that are not errors.
                # The most common is "'variables' has been replaced by 'variable' in lean 4"
                #
                # if not ("declaration uses 'sorry'" in m['data'] or 'failed' in m['data']):
                #     raise ValueError(
                #         f"Unexpected warning: {m['data']}")
                # complete = False
                warnings.append(m)

            elif m['severity'] == 'info':
                infos.append(m)
            else:
                raise ValueError(
                    f"Unexpected severity: {m['severity']}")

        self.truncate(sorries, errors)
        self.get_goals(goals)
        if self.valid_code.strip() == "":
            # This means that the new code was truncated to nothing,
            # i.e. the first new line of code written caused an error.
            # This is a dead state.
            self._dead = True
            self._win = False
            return

        if complete:
            print("Marked as complete!")
            print(repl_result)
            self._dead = False
            self._win = True
            return

        self._dead = False
        self._win = False

        return ()
