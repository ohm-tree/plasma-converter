import os
from dataclasses import dataclass
from typing import Dict, Sequence, TypedDict

import numpy as np
from wayfinder.games import *

from src.workers.worker import *

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 50000\nset_option linter.all false\nopen BigOperators Real Nat Topology Rat\n\n"


class ScratchpadStateError(Exception):
    pass


@dataclass(frozen=True)
class LeanMessage:
    """Represents a message from the Lean verifier"""
    severity: str  # 'error', 'warning', 'info'
    data: str
    line: int  # Line number (1-indexed)
    col: int  # Column number (0-indexed)


def to_annotated_string(code: str, messages: tuple[LeanMessage]) -> str:
    """Convert the code and annotations into a single string with inline comments"""
    lines = code.split('\n')
    sorted_annotations = sorted(
        messages, key=lambda x: x.line)
    res = ""
    annotation_index = 0
    for i, line in enumerate(lines):
        res += line + '\n'
        # lines are 1-indexed
        while annotation_index < len(sorted_annotations) and sorted_annotations[annotation_index].line - 1 == i:
            message = sorted_annotations[annotation_index]
            res += f"<{message.severity}>" + '\n'
            res += message.data + '\n'
            res += f"</{message.severity}>" + '\n'
            annotation_index += 1
    return res


@dataclass(frozen=True)
class ScratchpadMove:
    """Represents a move in the game"""
    new_code: str  # Complete new Lean code
    scratchpad_append: str  # Text to append to scratchpad


@dataclass(frozen=True)
class ScratchpadState(State):
    code: str  # Raw Lean code
    depth: int
    dead: bool
    messages: tuple[LeanMessage]  # All messages from verifier
    annotated_code: str
    scratchpad: str  # Arbitrary text for notes/working

    @classmethod
    def saves(cls, states: Sequence['ScratchpadState'], filename: str) -> None:
        """Save a collection of states to a file."""
        np.save(filename, [{
            'code': state.code,
            'depth': state.depth,
            'dead': state.dead,
            'messages': state.messages,
            'annotated_code': state.annotated_code,
            'scratchpad': state.scratchpad
        } for state in states])

    @classmethod
    def loads(cls, filename: str) -> list['ScratchpadState']:
        """Load a collection of states from a file."""
        states = np.load(filename, allow_pickle=True)
        return [cls(**state) for state in states]


def truncate_middle(text: str, max_length: int) -> str:
    """
    Truncates a string from the middle if it exceeds max_length.
    Adds an ellipsis message in the middle of truncated text.
    """
    if len(text) <= max_length:
        return text

    ellipsis = " <<Message is too long, truncated middle>> "
    available_chars = max_length - len(ellipsis)
    prefix_length = available_chars // 2
    suffix_length = available_chars - prefix_length

    return text[:prefix_length] + ellipsis + text[-suffix_length:]


class ProcessResult(TypedDict):
    """Result from processing Lean verifier response"""
    messages: tuple[LeanMessage, ...]  # Processed and truncated messages
    dead: bool  # Whether this is a dead state
    win: bool  # Whether this is a winning state


class ScratchpadGame(Game[ScratchpadMove, ScratchpadState]):
    """
    The ScratchpadGame class implements a game interface for interacting with the Lean 4 proof assistant.
    This is the "core" of the game, and does not involve implementation details which
    should be delegated to a higher-level algorithm for playing the game.

    The game state consists of:
    1. The current Lean code
    2. Messages (errors, warnings, etc.) from the Lean verifier
    3. Annotated code showing where errors/warnings occur
    4. A scratchpad for arbitrary notes

    A move consists of:
    1. A complete new Lean code that replaces the current code
    2. Text to append to the scratchpad

    The game terminates when either:
    1. The code is correct (victory, reward +1)
    2. The maximum depth is reached (loss, reward -1)
    3. The state is marked as dead (loss, reward -1)
    """

    def __init__(self,
                 worker: Worker,
                 informal_problem: str,
                 problem: str,
                 tactic_state: str,  # Unused
                 natural_language_proof: str,
                 header: str = LEAN4_DEFAULT_HEADER,
                 max_depth: int = 40,
                 max_message_length: int = 512,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.worker = worker

        self.problem: str = problem
        self.informal_problem: str = informal_problem
        self.natural_language_proof: str = natural_language_proof
        self.header: str = header
        self.max_depth: int = max_depth

        self.max_message_length: int = max_message_length

        self._win = {}

    ################################################################
    # Implementing the Game Interface
    ################################################################

    @property
    def death_value(self):
        return -1.0

    async def starting_state(self, **kwargs) -> 'ScratchpadState':
        """
        Returns the starting state of the game.
        """
        res = ScratchpadState(
            code=self.problem,
            depth=0,
            dead=False,
            messages=tuple(),
            annotated_code=to_annotated_string(
                self.problem, tuple()),
            scratchpad=""
        )
        self._win[res] = False
        return res

    async def is_legal(self, state: ScratchpadState, action: ScratchpadMove) -> bool:
        return True

    async def next_state(self, state: ScratchpadState, action: ScratchpadMove) -> 'ScratchpadState':
        """
        Returns the next state of the game given a current state and action.

        The new state will contain:
        - The new code from the action
        - Messages from the Lean verifier about this code
        - Annotated version of the code showing error locations
        - Updated scratchpad with the action's append text

        Parameters
        ----------
        state: ScratchpadState
            The current state of the game.
        action: ScratchpadMove
            The action containing new code and scratchpad text.

        Raises
        ------
        ScratchpadStateError
            If the current state is terminal
        """
        if await self.terminal(state):
            raise ScratchpadStateError(
                "Cannot get the next state of a terminal ScratchpadState.")

        new_code = self.problem + action.new_code

        response = await self.worker.query(
            task={
                'task': new_code,
                'channel': self.worker.name
            },
            channel='lean'
        )
        repl_result = response['response']
        process_result = self.process(repl_result)

        new_state = ScratchpadState(
            code=new_code,
            depth=state.depth + 1,
            dead=process_result['dead'],
            messages=tuple(process_result['messages']),
            annotated_code=to_annotated_string(
                new_code, process_result['messages']),
            scratchpad=state.scratchpad +
            action.scratchpad_append.strip('\n') + '\n'
        )

        self._win[new_state] = process_result['win']

        return new_state

    async def terminal(self, state: ScratchpadState) -> bool:
        return state.dead or state.depth >= self.max_depth or self._win[state]

    async def reward(self, state: ScratchpadState) -> float:
        """
        Returns a float consisting of the reward for the player.
        """
        if not await self.terminal(state):
            raise ScratchpadStateError(
                "Cannot get the reward of a non-terminal ScratchpadState.")
        return 1.0 if self._win[state] else -1.0

    async def victorious(self, state: ScratchpadState) -> bool:
        if not await self.terminal(state):
            raise ScratchpadStateError(
                "Cannot check if a non-terminal ScratchpadState is victorious.")
        return self._win[state]

    async def make_root(self, state: ScratchpadState) -> None:
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

    def process(self, repl_result: dict) -> ProcessResult:
        """
        Process the Lean verifier response into a structured format.

        Converts raw messages into LeanMessage objects, truncating long messages
        from the middle. Determines if the code is complete (no errors/sorries)
        and collects all relevant messages.

        Parameters
        ----------
        repl_result: dict
            Raw response from the Lean verifier

        Returns
        -------
        ProcessResult
            Contains processed messages and state information
        """
        messages = []
        complete = True

        # Convert raw messages to LeanMessage objects
        for m in repl_result.get('messages', []):
            # Truncate message data if too long
            truncated_data = truncate_middle(
                m['data'], self.max_message_length)

            msg = LeanMessage(
                severity=m['severity'],
                data=truncated_data,
                line=m['pos']['line'],
                col=m['pos']['column']
            )

            if msg.severity == 'error':
                complete = False
                messages.append(msg)
            elif msg.severity == 'warning':
                if "declaration uses 'sorry'" in msg.data or 'failed' in msg.data:
                    complete = False
                messages.append(msg)
            elif msg.severity == 'info':
                messages.append(msg)

        return {
            'messages': tuple(messages),
            'dead': False,
            'win': complete
        }

    def pretty_print(self, state: ScratchpadState) -> str:
        """
        Pretty print the current state of the game.

        Returns a string containing:
        1. The standard Lean header
        2. The problem statement
        3. The current annotated code
        4. The scratchpad content
        """
        res = ""
        res += self.header.strip('\n') + '\n'
        res += state.annotated_code.strip('\n') + '\n'
        res += "Scratchpad:\n"
        res += state.scratchpad.strip('\n') + '\n'

        return res


async def test_lean_game():
    class FakeWorker():
        def __init__(self):
            self.name = "fake"
            self.responses = {
                # Hardcoded responses for testing
                "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n": {
                    'response': {"messages":
                                 [{"severity": "error",
                                   "pos": {"line": 2, "column": 0},
                                     "endPos": None,
                                     "data": "unexpected end of input; expected '{'"},
                                     {"severity": "error",
                                      "pos": {"line": 1, "column": 157},
                                      "endPos": {"line": 1, "column": 159},
                                      "data":
                                      "unsolved goals\na r : ℝ\nu : ℕ → ℝ\nh₀ : ∀ (k : ℕ), u k = a * r ^ k\nh₁ : u 1 = 2\nh₂ : u 3 = 6\n⊢ u 0 = 2 / √3 ∨ u 0 = -(2 / √3)"}],
                                 "env": 1,
                                 "ast": []}
                },
                "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero, Nat.succ_add]\n": {
                    'response': {"messages":
                                 [{"severity": "error",
                                   "pos": {"line": 1, "column": 157},
                                     "endPos": {"line": 2, "column": 101},
                                     "data":
                                     "unsolved goals\na r : ℝ\nu : ℕ → ℝ\nh₀ : ∀ (k : ℕ), u k = a * r ^ k\nh₁ : a * r ^ succ 0 = 2\nh₂ : a * r ^ 3 = 6\n⊢ a * r ^ 0 = 2 / √3 ∨ a * r ^ 0 = -(2 / √3)"}],
                                 "env": 2,
                                 "ast": []}
                }
            }

        async def query(self, task: dict, channel: str) -> dict:
            return self.responses[task['task']]

    EXAMPLE_PROBLEM = "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\n"

    worker = FakeWorker()
    game = ScratchpadGame(
        worker=worker,
        problem=EXAMPLE_PROBLEM,
        header=LEAN4_DEFAULT_HEADER
    )
    state = await game.starting_state()
    print(game.pretty_print(state))

    state = await game.next_state(state, ScratchpadMove(
        new_code="  simp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero, Nat.succ_add]\n",
        scratchpad_append="This is a scratchpad message\n"
    ))
    print(game.pretty_print(state))

    state = await game.next_state(state, ScratchpadMove(
        new_code="",
        scratchpad_append="This is a scratchpad message\n"
    ))
    print(game.pretty_print(state))


if __name__ == "__main__":
    asyncio.run(test_lean_game())


"""
{"cmd": "theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2) (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by\nsimp_all only [Nat.one_eq_succ_zero, Nat.zero_eq, zero_add, Nat.add_succ, Nat.add_zero, Nat.succ_add]\n", "env": 0}

"""
