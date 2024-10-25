import os
from dataclasses import dataclass

import numpy as np
from wayfinder.games import *

from src.workers.worker import *

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

"""
10/25/2024 Breakthrough: Setting maxHeartbeats here allows me to make the Lean 4 kernel terminate due to an internal timeout!
This is really strong; previously it was set to 0 (to allow unlimited computation); this is good for a
"no-holds-barred" run, but in terms of allowing for reasonable throughput compliant with low pexpect timeouts, small
numbers are really good.

The Lean default is 200000, which means roughly that each command is upper-bounded by a couple of seconds.
Roughly 10000 means that each command runs in sub-seconds, which is what we want.

The upshot: we should expect to see almost no timeout errors in the future!
"""

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 10000\nset_option linter.all false\nopen BigOperators Real Nat Topology Rat\n\n"

"""
Old header:
LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\nset_option linter.all false\nopen BigOperators Real Nat Topology Rat\n\n"

"""


class LeanStateError(Exception):
    pass


@dataclass(frozen=True)  # Hashable
class LeanMove:
    new_code: str


@dataclass(frozen=True)
class LeanState(State):
    code: str
    depth: int
    tactic_state: str
    dead: bool

    @classmethod
    def saves(cls, states: list['LeanState'], filename: str):
        """
        Save a collection of states to a file.
        """
        np.save(filename, [{
            'code': state.code,
            'depth': state.depth,
            'tactic_state': state.tactic_state,
            'dead': state.dead
        } for state in states])

    @classmethod
    def loads(cls, filename: str) -> list['LeanState']:
        """
        Load a collection of states from a file.
        """
        states = np.load(filename, allow_pickle=True)
        return [cls(**state) for state in states]


class LeanGame(Game[LeanMove, LeanState]):
    """
    The LeanState class implements the game state for a Lean 4 proof assistant game.
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

    def __init__(self,
                 worker: Worker,
                 problem: str,
                 tactic_state: str,
                 header: str = LEAN4_DEFAULT_HEADER,
                 max_depth: int = 40,
                 max_tactic_state_length: int = 512,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.worker = worker

        self.problem: str = problem
        self.header: str = header
        self.max_depth: int = max_depth

        self.max_tactic_state_length: int = max_tactic_state_length

        self.root_tactic_state: str = tactic_state

        self._win = {}

    ################################################################
    # Implementing the Game Interface
    ################################################################

    @property
    def death_value(self):
        return -1.0

    async def starting_state(self) -> 'LeanState':
        """
        Returns the starting state of the game.
        """
        res = LeanState(
            code="",
            depth=0,
            tactic_state=self.root_tactic_state,
            dead=False
        )
        self._win[res] = False
        return res

    async def is_legal(self, state: LeanState, action: LeanMove) -> bool:
        return True

    async def next_state(self, state: LeanState, action: LeanMove) -> 'LeanState':
        """
        Returns the next state of the game given a current state and action.
        Requires that the state is non-terminal and action is legal.

        This method cannot be called if the state is terminal!

        Parameters
        ----------
        state: LeanState
            The current state of the game.
        action: LeanMove
            The action to be taken.
        """
        if await self.terminal(state):
            raise LeanStateError(
                "Cannot get the next state of a terminal LeanState.")

        response = await self.worker.query(
            task={
                'task': self.problem + state.code + action.new_code,
                'channel': self.worker.name
            },
            channel='lean'
        )
        repl_result = response['response']
        process_result = self.process(state.code, action.new_code, repl_result)

        new_state = LeanState(
            code=state.code + process_result['valid_code'],
            depth=state.depth + 1,
            tactic_state=process_result['tactic_state'],
            dead=process_result['dead']
        )

        self._win[new_state] = process_result['win']

        return new_state

    async def terminal(self, state: LeanState) -> bool:
        return state.dead or state.depth >= self.max_depth or self._win[state]

    async def reward(self, state: LeanState) -> float:
        """
        Returns a float consisting of the reward for the player.
        """
        if not await self.terminal(state):
            raise LeanStateError(
                "Cannot get the reward of a non-terminal LeanState.")
        return 1.0 if self._win[state] else -1.0

    async def victorious(self, state: LeanState) -> bool:
        if not await self.terminal(state):
            raise LeanStateError(
                "Cannot check if a non-terminal LeanState is victorious.")
        return self._win[state]

    async def make_root(self, state: LeanState) -> None:
        """
        Severs any references to the parent of
        this state, making it the root of the tree.
        """
        pass

    async def clear_cache(self) -> None:
        """
        Clear the cache of the game.
        """
        self._win = {}
        self._dead = {}
        super().clear_cache()

    ################################################################
    # Helper Functions
    ################################################################

    def get_index(self, s: str, row: int, col: int) -> int:
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
        if col > line_lengths[row-1]:
            # The col CAN be exactly line_lengths; this is just the format
            # of the lean kernel's outputs. in this case, we will truncate
            # following the newline.
            raise ValueError(
                f"Column is too large. row = {row}, col = {col}, line_lengths = {line_lengths}, s = {s}")
        return sum(line_lengths[:row-1]) + col

    def truncate(self, old_code: str, new_code: str, sorries: list[dict], errors: list[dict]) -> str:
        """
        First, we need to find the last valid tactic.

        Unsolved goals also show up as errors, and we ignore them;
        they have been omitted from the errors list in post_process()
        already.

        Parameters
        ----------
        sorries: list[dict]
            A list of sorry messages.
        errors: list[dict]
            A list of error messages.

        """
        code_no_header = self.problem + old_code + new_code

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

        valid_code = code_no_header[:truncate_pos]
        assert valid_code.startswith(self.problem)
        valid_code = valid_code[len(self.problem):]
        # I guess this might not be true in some edge cases?
        # assert valid_code.startswith(self.old_code)
        valid_code = valid_code[len(old_code):]

        return valid_code

    def get_goals(self, goals: list[dict]) -> str:
        """
        Get the the most recent goal from the Lean 4 verification.

        Parameters
        ----------
        goals: list[dict]
            A list of goal messages.
        """

        tactic_state = ""
        max_line = float('-inf')
        max_column = float('-inf')
        for tactic in goals:
            if (tactic['pos']['line'] > max_line) or (tactic['pos']['line'] == max_line and tactic['pos']['column'] > max_column):
                max_line = tactic['pos']['line']
                max_column = tactic['pos']['column']
                tactic_state = tactic['data'].lstrip()[
                    len("unsolved goals\n"):]

        if len(tactic_state) > self.max_tactic_state_length:
            # If too long, truncate the middle section, and replace with the messages "<<Message is too long, truncated middle>>"
            error_message = " <<Message is too long, truncated middle>> "
            available_characters = self.max_tactic_state_length - \
                len(error_message)
            prefix_length = available_characters // 2
            suffix_length = available_characters - prefix_length

            tactic_state = tactic_state[:prefix_length] + \
                error_message + tactic_state[-suffix_length:]
        return tactic_state

    def process(self, old_code: str, new_code: str, repl_result: dict[str, list[dict[str, str]]]):
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

        dead_result = {
            'tactic_state': "",
            'valid_code': "",
            'dead': True,
            'win': False
        }

        if 'system_error' in repl_result or 'message' in repl_result or ('ast' not in repl_result):
            return dead_result

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
        complete = True

        if len(sorries) > 0:
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

        valid_code = self.truncate(old_code, new_code, sorries, errors)

        if valid_code.strip() == "":
            # This means that the new code was truncated to nothing,
            # i.e. the first new line of code written caused an error.
            # This is a dead state.
            return dead_result

        tactic_state = self.get_goals(goals)
        if complete:
            return {
                'tactic_state': tactic_state,
                'valid_code': valid_code,
                'dead': False,
                'win': True
            }

        return {
            'tactic_state': tactic_state,
            'valid_code': valid_code,
            'dead': False,
            'win': False
        }

    def pretty_print(self, state: LeanState) -> str:
        """
        Pretty print the current state of the game as valid Lean 4 code.
        """
        res = ""
        res += self.header
        res += self.problem
        res += state.code
        # Add some metadata in a comment.
        # res += f"\n\n-- Depth: {state.depth}\n"
        return res
