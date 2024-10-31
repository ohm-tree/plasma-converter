import os
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from wayfinder.games import *

from src.workers.worker import *

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 50000\nset_option linter.all false\nopen BigOperators Real Nat Topology Rat\n\n"


class LeanStateError(Exception):
    pass


@dataclass
class LeanMessage:
    """Represents a message from the Lean verifier"""
    severity: str  # 'error', 'warning', 'info'
    data: str
    pos: dict[str, int]  # line and column information


@dataclass
class AnnotatedCode:
    """Represents Lean code with inline annotations"""
    raw_code: str
    annotations: list[tuple[int, str]]  # (position, annotation message)

    def to_annotated_string(self) -> str:
        """Convert the code and annotations into a single string with inline comments"""
        lines = self.raw_code.split('\n')
        # Sort annotations by line number in reverse order to avoid position shifts
        sorted_annotations = sorted(
            self.annotations, key=lambda x: x[0], reverse=True)

        for pos, message in sorted_annotations:
            line_num = self.raw_code.count('\n', 0, pos)
            lines[line_num] += f"  -- {message}"

        return '\n'.join(lines)


@dataclass(frozen=True)
class LeanMove:
    """Represents a move in the game"""
    new_code: str  # Complete new Lean code
    scratchpad_append: str  # Text to append to scratchpad


@dataclass(frozen=True)
class LeanState(State):
    code: str  # Raw Lean code
    depth: int
    dead: bool
    messages: list[LeanMessage]  # All messages from verifier
    annotated_code: AnnotatedCode
    scratchpad: str  # Arbitrary text for notes/working

    @classmethod
    def saves(cls, states: Sequence['LeanState'], filename: str) -> None:
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
    def loads(cls, filename: str) -> list['LeanState']:
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


class LeanGame(Game[LeanMove, LeanState]):
    """
    The LeanGame class implements a game interface for interacting with the Lean 4 proof assistant.
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
                 problem: str,
                 header: str = LEAN4_DEFAULT_HEADER,
                 max_depth: int = 40,
                 max_message_length: int = 512,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.worker = worker

        self.problem: str = problem
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

    async def starting_state(self) -> 'LeanState':
        """
        Returns the starting state of the game.
        """
        res = LeanState(
            code="",
            depth=0,
            dead=False,
            messages=[],
            annotated_code=AnnotatedCode(raw_code="", annotations=[]),
            scratchpad=""
        )
        self._win[res] = False
        return res

    async def is_legal(self, state: LeanState, action: LeanMove) -> bool:
        return True

    async def next_state(self, state: LeanState, action: LeanMove) -> 'LeanState':
        """
        Returns the next state of the game given a current state and action.

        The new state will contain:
        - The new code from the action
        - Messages from the Lean verifier about this code
        - Annotated version of the code showing error locations
        - Updated scratchpad with the action's append text

        Parameters
        ----------
        state: LeanState
            The current state of the game.
        action: LeanMove
            The action containing new code and scratchpad text.

        Raises
        ------
        LeanStateError
            If the current state is terminal
        """
        if await self.terminal(state):
            raise LeanStateError(
                "Cannot get the next state of a terminal LeanState.")

        response = await self.worker.query(
            task={
                'task': self.problem + action.new_code,
                'channel': self.worker.name
            },
            channel='lean'
        )
        repl_result = response['response']
        process_result = self.process(repl_result)

        # Create annotations from messages
        annotations = []
        for msg in process_result['messages']:
            pos = self.get_index(action.new_code,
                                 msg.pos['line'],
                                 msg.pos['column'])
            annotations.append((pos, f"{msg.severity}: {msg.data}"))

        new_state = LeanState(
            code=action.new_code,
            depth=state.depth + 1,
            dead=process_result['dead'],
            messages=process_result['messages'],
            annotated_code=AnnotatedCode(
                raw_code=action.new_code,
                annotations=annotations
            ),
            scratchpad=state.scratchpad + action.scratchpad_append
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

    def process(self, repl_result: dict) -> dict:
        """
        Process the Lean verifier response into a structured format.

        Converts raw messages into LeanMessage objects, truncating long messages
        from the middle. Determines if the code is complete (no errors/sorries)
        and collects all relevant messages.

        Parameters
        ----------
        code: str
            The Lean code being processed
        repl_result: dict
            Raw response from the Lean verifier

        Returns
        -------
        dict
            Contains:
            - messages: list[LeanMessage] - Processed and truncated messages
            - dead: bool - Whether this is a dead state
            - win: bool - Whether this is a winning state
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
                pos=m['pos']
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
            'messages': messages,
            'dead': False,
            'win': complete
        }

    def pretty_print(self, state: LeanState) -> str:
        """
        Pretty print the current state of the game.

        Returns a string containing:
        1. The standard Lean header
        2. The problem statement
        3. The current code
        4. The scratchpad content (as Lean comments)

        Note: The code portion is raw (without annotations)
        so it can be used with the Lean verifier. The scratchpad
        is included as comments at the end.
        """
        res = ""
        res += self.header
        res += self.problem
        res += state.code

        # Add scratchpad as comments if non-empty
        if state.scratchpad.strip():
            res += "\n\n/-\nScratchpad:\n"
            res += state.scratchpad
            res += "\n-/\n"

        return res
