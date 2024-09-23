import json
import os
import subprocess
import tempfile
import time
import traceback
from pprint import pprint
from typing import Callable, List, Optional

import numpy as np
import pexpect
import torch.nn as nn

from src.games.ast_parser import lean4_parser
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


def process_lean4_results(
    code: str,
    repl_result: dict,
):
    start_time = time.time()
    system_messages = ''
    try:
        ast_results = lean4_parser(
            code, repl_result['ast']) if 'ast' in repl_result and repl_result['ast'] else {}
        result = {
            "sorries": repl_result.get('sorries', []),
            "tactics": repl_result.get('tactics', []),
            "errors": [m for m in repl_result.get('messages', []) if m['severity'] == 'error'],
            "warnings": [m for m in repl_result.get('messages', []) if m['severity'] == 'warning'],
            "infos": [m for m in repl_result.get('messages', []) if m['severity'] == 'info'],
            "system_messages": system_messages,
            "system_errors": None,
            "ast": ast_results,
            "verified_code": code,
            "env": repl_result.get('env', None),
        }
        result['pass'] = not result['errors']
        result['complete'] = result['pass'] and not result['sorries'] and not any(
            "declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
    except:
        result = {
            "pass": False,
            "complete": False,
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages
        }
    result['verify_time'] = time.time() - start_time
    return result


def segmentation(
    code: str,
    formal_statement: str,
    result: dict,
    header: str = LEAN4_DEFAULT_HEADER,
    tailer: str = "",
):

    if result['system_errors'] is not None:
        print("System errors, returning empty segments")
        # compiler timeout
        return []

    full_code = ''.join(
        [header, formal_statement, code.rstrip(' \n'), tailer]
    )

    _full_code_lines = full_code.split('\n')
    _line_offset, _offset = [], -1
    for _line in _full_code_lines:
        _offset += 1  # '\n'
        _line_offset.append(_offset)
        _offset += len(_line)

    def _get_idx(pos_info):
        """
        Convert a (line, column) dict to the index of the character in the full code.
        """
        return _line_offset[pos_info['line'] - 1] + pos_info['column']

    """
    First, we need to find the last valid tactic.

    Unsolved goals also show up as errors, and we ignore them.
    """

    _prefix_len = len(header) + len(formal_statement)
    truncate_pos = len(full_code) - len(tailer)
    for info in result['sorries'] + result['errors']:
        if info.get('data', str()).lstrip().startswith('unsolved goals'):
            continue
        info_pos = _get_idx(info['pos'])
        # if info_pos >= _prefix_len:
        truncate_pos = min(truncate_pos, info_pos)

    if truncate_pos <= _prefix_len:
        # all proof lines are invalid
        return []

    partial_code = full_code[:truncate_pos]

    code_lines = partial_code.split('\n')
    pos_last, segments = _prefix_len, []
    # pos_last, segments = 0, []
    for line_idx in range(len(code_lines)):
        # I want to add segments for each line, including
        # the problem statement etc.

        if _line_offset[line_idx] < _prefix_len:
            continue

        def compute_last_valid_char_pos(line):
            idx, last_non_blank = 0, len(line) + 1
            while idx < len(line):
                if line[idx: idx+2] == '--':
                    return last_non_blank
                elif line[idx: idx+2] == '/-':
                    if '-/' not in line[idx+2:]:
                        # cannot split in this line
                        return len(line) + 1
                    idx = line.find('-/', idx+2) + 1
                elif line[idx] != ' ':
                    last_non_blank = idx
                idx += 1
            return last_non_blank

        line_lastChar = _line_offset[line_idx] + \
            compute_last_valid_char_pos(code_lines[line_idx])
        line_endPos = _line_offset[line_idx] + \
            len(code_lines[line_idx])

        pos_min, goal = 1e9, None
        for tactic_info in result['ast']['tactics']:
            pos, endPos = tactic_info['pos'], tactic_info['endPos']
            if line_lastChar <= endPos and endPos <= line_endPos and pos < pos_min:
                pos_min = pos
                goal = tactic_info['stateAfter']
        if goal is None:
            continue

        for tactic_info in result['ast']['tactics']:
            pos, endPos = tactic_info['pos'], tactic_info['endPos']
            if pos_last < endPos and endPos <= line_endPos and pos < pos_min:
                pos_min = pos

        while pos_min > 0 and partial_code[pos_min - 1] != '\n':
            pos_min -= 1
        indent_len = 0
        while partial_code[pos_min + indent_len] == ' ':
            indent_len += 1
        newline_with_indent = '\n' + ' ' * indent_len

        segments.append(AttrDict(
            tactic_code=partial_code[pos_last: line_endPos] + '\n',
            state_comment=newline_with_indent.join([
                ' ' * indent_len + '/- tactic state:',
                '  ' + goal.replace('\n',
                                    newline_with_indent + '  '),
                '-/\n'
            ]),
            goal=goal,
            indent=indent_len,
        ))
        pos_last = line_endPos + 1
    if result['complete'] and (len(segments) == 0 or segments[-1].goal != 'no goals' or segments[-1].indent != segments[0].indent):
        indent_len = 2 if len(segments) == 0 else segments[0].indent
        newline_with_indent = '\n' + ' ' * indent_len
        segments.append(AttrDict(
            tactic_code=partial_code[pos_last:].rstrip(' \n') + '\n',
            state_comment=newline_with_indent.join([
                ' ' * indent_len + '/- tactic state:',
                '  no goals',
                '-/\n'
            ]),
            goal='no goals',
            indent=indent_len,
        ))
    segments = [seg for seg in segments if len(
        seg.tactic_code.strip(' \n')) > 0]
    return segments


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
                 tailer: str = "",
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
        | tailer           | Required         | Required     | Required        |
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
        tailer: str
            The tailer for the Lean code.
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
        self.tailer: str = tailer
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
        
        def fancy_field(name: str, value: str, length = 80, tick = '-') -> str:
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
            res += fancy_field("Completed Rollout without Truncation", self.new_code)
        else:
            res += fancy_field("[New Code will be here]", "\n")
        
        res += fancy_field("Tailer", self.tailer)
        res += fancy_field("Old Tactic State", self.old_tactic_state)
        if self.processed:
            res += fancy_field("New Tactic State", self.tactic_state)
        
        res += fancy_field("Meta", f"Processed: {self.processed}, Rollout Done: {self.rollout_done}\n"\
            f"Win: {self.win}, Dead: {self.dead}\n"\
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
            "tailer": self.tailer,
            "rollout_done": self.rollout_done,
            "processed": self.processed,
            "new_code": self.new_code,
            "win": self.win,
            "tactic_state": self.tactic_state,
            "valid_code": self.valid_code,
        }

    def __str__(self) -> str:
        return f"LeanGameState({self.problem}, {self.code}, processed = {self.processed})"

    def post_LLM_rollout(self, new_code: str):
        """
        This function is called after the LLM rollout is done.
        """
        if self.rollout_done:
            raise LeanGameStateError(
                "Should not LLM-post-process a LeanGameState that has already had an LLM rollout.")
        self.new_code = new_code
        self.rollout_done = True

    def pre_process(self) -> str:
        """
        This function is called before the state is processed.
        It prepares a string query for the lean 4 verifier.
        """

        if self.processed:
            raise LeanGameStateError(
                "Should not pre-process a LeanGameState that has already been processed.")

        # full_code without the header.
        return ''.join(
            [self.problem, self.old_code, self.comment, self.new_code, self.tailer]
        )

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
        full_code = ''.join(
            [self.header, self.problem, self.old_code, self.comment, self.new_code, self.tailer]
        )

        print("Full code:")
        print(full_code)

        result = process_lean4_results(
            full_code,
            repl_result,
        )
        print("Result:")
        pprint(result)

        segments = segmentation(
            ''.join([self.old_code, self.comment, self.new_code]),
            self.problem,
            result,
            self.header,
            self.tailer,
        )

        print("Segments:")
        pprint(segments)

        self.new_code = ""
        i = 0
        while len(self.new_code) <= len(self.old_code) and i < len(segments):
            self.new_code += segments[i].tactic_code
            i += 1
        i -= 1

        print("New code:")
        print(self.new_code)
        assert self.new_code.startswith(self.old_code)

        if 0 <= i < len(segments):
            self.tactic_state = segments[i].state_comment

        if result['complete']:
            self.dead = False
            self.win = True
            return

        # After 480 mod (done) this should never be triggered
        if self.tactic_state is None:
            # Special case, root node.
            print(
                "BAD should not be here because we initialize and pass down tactic stat")
            s = result['errors'][1]['data']
            tactic_state = s[s.find("\n"):]
            self.tactic_state = '/- tactic state:\n' + tactic_state + '\n-/\n'

        if self.old_tactic_state == self.tactic_state and self.old_code != "":
            print("Inside post_process, found new code = old code thus dead")
            self.dead = True
            self.win = False
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
            [self.old_code, self.comment, self.new_code]
        )

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



class LeanGame(Game[LeanGameState]):
    """
    The LeanGame class implements the game logic
    for a Lean 4 proof assistant game.
    """

    def __init__(self,
                 comment_seeds: List[str],
                 max_depth: Optional[int] = 10,
                 max_completion_len: int = 2000):
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
                    tailer: str = "",
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
        tailer: str
            The tailer for the Lean code.
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
            tailer=tailer,
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

        return LeanGameState(
            problem=state.problem,
            old_code=state.code(),
            old_tactic_state=state.tactic_state,
            comment=comment,
            depth=state.depth + 1,
            header=state.header,
            tailer=state.tailer,
            rollout_done=False,
            processed=False,
        )

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
