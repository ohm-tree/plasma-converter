from __future__ import annotations

import os
import random
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from src.games.game import Game
from src.games.lean_game import LeanGame, LeanGameState, LeanGameStateStep
from src.uct.uct_node import UCTNode

if TYPE_CHECKING:
    from src.workers.mcts_inference_worker import (
        CompletionTaskType,
        LeanTaskType,
        MCTSWorker,
        MCTSWorkerType,
        PolicyValueTaskType,
    )
    from src.workers.worker import TaskIdentifier, TaskType, WorkerResponse

HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = 'mathlib4/'

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\nset_option linter.all false\nopen BigOperators Real Nat Topology Rat\n\n"


class AttrDict(dict):
    """A dictionary that allows attribute-style access."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class LeanGameStateError(Exception):
    pass


class LeanGameStateStep(IntEnum):
    INITIALIZED = 0
    ROLLOUT = 1
    PROCESSED = 2
    COMMENTED = 3


class LeanGameState:
    def __init__(self,
                 step: LeanGameStateStep,
                 problem: str,
                 old_code: str,
                 old_tactic_state: str,
                 comment: str,
                 depth: int,
                 header: str = LEAN4_DEFAULT_HEADER,
                 new_code: Optional[str] = None,
                 win: Optional[bool] = None,
                 dead: Optional[bool] = None,
                 tactic_state: Optional[str] = None,
                 valid_code: Optional[str] = None,
                 gen_comments: Optional[List[str]] = None,
                 ):
        """
        A LeanGameState can be in one of four states:
        1. Just-initialized. In this case, neither the LLM
        rollout nor the Lean verification has been run.
        2. Post-rollout. In this case, the LLM rollout has
        been run, but the Lean verification has not been run.
        3. Processed. In this case, both the LLM
        rollout and the Lean verification have been run.
        4. Commented. In this case, the state has been
        processed and the possible comments have been generated.

        Here are the fields and their statuses in each state:

        | Field            | Just-initialized | Post-rollout | Processed   | Commented   |
        | ---------------- | ---------------- | ------------ | ----------- | ----------- |
        | step             | "initialized"    | "rollout"    | "processed" | "commented" |
        | problem          | Required         | Required     | Required    | Required    |
        | old_code         | Required         | Required     | Required    | Required    |
        | old_tactic_state | Required         | Required     | Required    | Required    |
        | comment          | Required         | Required     | Required    | Required    |
        | depth            | Required         | Required     | Required    | Required    |
        | header           | Required         | Required     | Required    | Required    |
        | new_code         | None             | Required     | Required    | Required    |
        | win              | None             | None         | Required    | Required    |
        | dead             | None             | None         | Required    | Required    |
        | tactic_state     | None             | None         | Required    | Required    |
        | valid_code       | None             | None         | Required    | Required    |
        | gen_comments     | None             | None         | None        | Required    |

        Attempting to call methods like LeanGame.is_terminal(),
        Leangame.reward(), or LeanGame.next_state() on a
        non-fully-processed state will raise a LeanGameStateError.

        Parameters
        ----------
        step: LeanGameStateStep
            The step of the game state.
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
        gen_comments: Optional[List[str]]
            The comments that were generated by the LLM.
            This will be None if the proof is not fully processed.
        """
        self.step = step
        self.problem: str = problem
        self.old_code: str = old_code
        self.old_tactic_state: str = old_tactic_state
        self.comment: str = comment
        self.depth: int = depth
        self.header: str = header
        self.new_code: Optional[str] = new_code
        self.win: Optional[bool] = win
        self.dead: Optional[bool] = dead
        self.tactic_state: Optional[str] = tactic_state
        self.valid_code: Optional[str] = valid_code
        self.gen_comments: Optional[List[str]] = gen_comments

        self.check_state()

        # We don't want to re-generate a child when we re-do an action,
        # so pointers to the children are stored here.
        self.children = {}

        # This is just a unique identifier for the state.
        self._id = hash(random.random() + time.time())

    def check_state(self) -> None:
        """
        Check that the state is valid.
        """
        if None in [self.problem, self.old_code, self.old_tactic_state, self.comment, self.depth, self.header]:
            raise ValueError("None is not allowed for any required fields.")

        if self.step >= LeanGameStateStep.ROLLOUT:
            if self.new_code is None:
                raise ValueError(
                    "new_code is required if step is rollout, processed, or commented.")
        else:
            if self.new_code is not None:
                raise ValueError(
                    "new_code must be None if step is initialized.")

        if self.step >= LeanGameStateStep.PROCESSED:
            if None in [self.win, self.dead, self.tactic_state, self.valid_code]:
                raise ValueError(
                    "win, dead, tactic_state, and valid_code are required if step is processed or commented.")
        else:
            if [self.win, self.dead, self.tactic_state, self.valid_code].count(None) != 4:
                raise ValueError(
                    "win, dead, tactic_state, and valid_code must be None if step is initialized or rollout.")

        if self.step >= LeanGameStateStep.COMMENTED:
            if self.gen_comments is None:
                raise ValueError(
                    "gen_comments is required if step is commented.")
        else:
            if self.gen_comments is not None:
                raise ValueError(
                    "gen_comments must be None if step is initialized, rollout, or processed.")

    def __hash__(self) -> int:
        """
        Hash the state based on the id.
        """
        return self._id

    def add_child(self, action: int, child: 'LeanGameState'):
        if not (self.step >= LeanGameStateStep.COMMENTED):
            raise LeanGameStateError(
                "Cannot add a child to a LeanGameState if it has not been processed.")
        self.children[action] = child

    def terminal(self) -> bool:
        """
        We should be able to check if we're terminal
        even if we haven't done the PV or comment generation
        yet; this is useful because I have a lot of while not state.terminal()
        loops in the code.
        """
        if not (self.step >= LeanGameStateStep.PROCESSED):
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

        if self.step == LeanGameStateStep.INITIALIZED:
            status_code = "Just initialized\n"
        elif self.step == LeanGameStateStep.ROLLOUT:
            status_code = "Rollout done\n"
        elif self.step == LeanGameStateStep.PROCESSED:
            status_code = "Fully processed\n"
        elif self.step == LeanGameStateStep.COMMENTED:
            status_code = "Commented\n"

        res += fancy_field("Status", status_code)
        res += fancy_field("Header", self.header)
        res += fancy_field("Problem", self.problem)
        res += fancy_field("Old Code", self.old_code)
        res += fancy_field("Comment", self.comment)
        if self.step == LeanGameStateStep.INITIALIZED:
            res += fancy_field("[New Code will be here]", "\n")
        elif self.step == LeanGameStateStep.ROLLOUT:
            res += fancy_field("Completed Rollout without Truncation",
                               self.new_code)
        if self.step == LeanGameStateStep.PROCESSED:
            res += fancy_field("Valid Truncation of New Code", self.valid_code)

        if self.step == LeanGameStateStep.COMMENTED:
            res += fancy_field("Generated Comments",
                               '\n'.join(self.gen_comments))

        res += fancy_field("Old Tactic State", self.old_tactic_state)
        if (self.step >= LeanGameStateStep.PROCESSED):
            res += fancy_field("New Tactic State", self.tactic_state)

        res += fancy_field("Meta", f"Win: {self.win}, Dead: {self.dead}\n"
                           f"Depth: {self.depth} Number of Children: {len(self.children)}\n")
        return res

    def human_json(self) -> dict:
        """
        Returns a JSON representation of the game state.
        This is meant to be stored alongside game data
        by the workers so that we can debug easier.

        This is also useful for sending LeanGameStates to processes;
        we probably don't want all the references floating around.
        """
        return {
            # "step": self.step,
            "problem": self.problem,
            "old_code": self.old_code,
            "old_tactic_state": self.old_tactic_state,
            "comment": self.comment,
            "depth": self.depth,
            "header": self.header,
            "new_code": self.new_code,
            "win": self.win,
            "dead": self.dead,
            "tactic_state": self.tactic_state,
            "valid_code": self.valid_code,
            "gen_comments": self.gen_comments
        }

    @classmethod
    def from_json(cls, json: dict) -> 'LeanGameState':
        """
        Returns a LeanGameState from a JSON representation.
        """
        return cls(**json)

    def __str__(self) -> str:
        return f"LeanGameState({self.problem}, {self.code}, step = {self.step})"

    def compute(self, worker: MCTSWorker, result: Optional[WorkerResponse] = None) -> None:
        """

        Things I need to return:
         - whether or not I can backup.
         - whether or not i can
        """
        # Load any results from the completion queue, lean queue, and context_queue.
        # and enqueue them all to the lean_queue and context_queue.
        if result is None:
            worker.enqueue_task(
                obj=self.pre_LLM_rollout(),
                task_idx=hash(self),
                task_type=CompletionTaskType
            )
        elif result.task_id.task_type == CompletionTaskType:
            self.post_LLM_rollout(result['output'])
            worker.enqueue_task(
                obj=self.pre_process(),
                task_idx=hash(self),
                task_type=LeanTaskType
            )
        elif result.task_id.task_type == LeanTaskType:
            self.post_process(result['result'])
            node.is_processed = True

            if node.is_terminal:
                # compute the value estimate of the player at the terminal leaf
                value_estimate: float = game.reward(self)
                # Immediately backup the value estimate along the path to the root
                node.backup(value_estimate)

                if value_estimate == 1.0:
                    worker.logger.info(
                        "We think we just won! Here was the lean output:")
                    worker.logger.info(result['result'])
                    victorious_death = True
                    winning_node = node

            else:
                # Enqueue the node to the context_queue.
                worker.enqueue_task(
                    obj=self.pre_comments(),
                    task_idx=hash(node),
                    task_type=PolicyValueTaskType
                )
                context_waiting[hash(node)] = (node, time.time())

        elif result.task_id.task_type == PolicyValueTaskType:
            # Find the node that requested this policy value.
            node, time_init = context_waiting.pop(result['task_id'])

            time_taken = time.time() - time_init

            worker.logger.info("Received context output, took " +
                               str(time_taken) + " seconds.")
            sum_context_time += time_taken
            total_context += 1

            # Update the node with the policy value.
            self: LeanGameState = node.game_state
            self.post_comments(result['task_output']['comments'])

            # first, we need to comment the state right away.
            worker.logger.info(
                f"Received policy: {result['task_output']['policy']}")
            worker.logger.info(
                f"Received value: {result['task_output']['value']}")
            policy = np.array(result['task_output']['policy'])
            node.expand(policy,
                        result['task_output']['value'], train)
            node.backup(result['task_output']['value'])
        else:
            raise ValueError("Unknown task type.")

    def pre_LLM_rollout(self) -> str:
        """
        This function is called before the LLM rollout is done.
        It generates a prompt for the LLM.
        """
        if self.step != LeanGameStateStep.INITIALIZED:
            raise LeanGameStateError(
                "Should not LLM-pre-process a LeanGameState that has already had an LLM rollout.")

        return 'Complete the following Lean 4 code.\nHere is a hint:\n' + self.comment.strip() + \
            '\nThe tactic state is:\n' + \
            self.old_tactic_state.strip()+'\n```lean\n' + self.header + self.problem + \
            self.old_code

    def post_LLM_rollout(self, new_code: str):
        """
        This function is called after the LLM rollout is done.
        """
        if self.step != LeanGameStateStep.INITIALIZED:
            raise LeanGameStateError(
                "Should not LLM-post-process a LeanGameState that has already had an LLM rollout.")
        if new_code.endswith('```'):
            new_code = new_code[:-3]
        self.new_code = new_code
        self.step = LeanGameStateStep.ROLLOUT

    def pre_process(self) -> str:
        """
        This function is called before the state is processed.
        It prepares a string query for the lean 4 verifier.
        """

        if self.step != LeanGameStateStep.ROLLOUT:
            raise LeanGameStateError(
                "Should not pre-process a LeanGameState that has already been processed.")

        return ''.join(
            [self.problem, self.old_code, self.new_code]
        )

    @ classmethod
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
        if self.step != LeanGameStateStep.ROLLOUT:
            raise LeanGameStateError(
                "Should not post-process a LeanGameState that has not been rolled out.")
        self.step = LeanGameStateStep.PROCESSED

        if 'system_error' in repl_result or 'message' in repl_result or ('ast' not in repl_result):
            self.tactic_state = ""
            self.valid_code = ""
            self.dead = True
            self.win = False
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
            self.dead = True
            self.win = False
            return

        if complete:
            print("Marked as complete!")
            print(repl_result)
            self.dead = False
            self.win = True
            return

        self.dead = False
        self.win = False

    def pre_comments(self) -> dict:
        """
        This function is called before the comments are generated.
        """
        if self.step != LeanGameStateStep.PROCESSED:
            raise LeanGameStateError(
                "Should not pre-comments a LeanGameState that has not been processed.")

        return self.human_json()

    def post_comments(self, gen_comments: List[str]):
        """
        This function is called after the comments are generated.
        """
        if self.step != LeanGameStateStep.PROCESSED:
            raise LeanGameStateError(
                "Should not post-comments a LeanGameState that has already been commented.")
        self.gen_comments = gen_comments
        self.step = LeanGameStateStep.COMMENTED

    def code(self) -> str:
        """
        Returns the full code of the state.
        old_code + comment + new_code

        Returns
        -------
        str
            The full code of the state.
        """
        if not (self.step >= LeanGameStateStep.PROCESSED):
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
                 num_comment_seeds: int,
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
        num_comment_seeds: int
            The number of comment seeds.
        max_depth: Optional[int]
            The maximum depth of the proof.
        max_completion_len: int
            The maximum length of the completion.
        """
        self.num_comment_seeds = num_comment_seeds
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
            step=LeanGameStateStep.PROCESSED,
            problem=problem,
            old_code="",
            old_tactic_state="",
            comment="",
            depth=0,
            header=header,
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

        Parameters
        ----------
        state: LeanGameState
            The current state of the game.
        action: int
            The action to be taken.
        """

        if action in state.children:
            # We've already computed this child, and we can return the cached result.
            return state.children[action]

        if not state.step >= LeanGameStateStep.COMMENTED:
            raise LeanGameStateError(
                "Cannot get the next state of a LeanGameState that has not been processed.")

        comment = state.gen_comments[action]

        if state.terminal():
            raise LeanGameStateError(
                "Cannot get the next state of a terminal LeanGameState.")

        new_state = LeanGameState(
            step=LeanGameStateStep.INITIALIZED,
            problem=state.problem,
            old_code=state.code(),
            old_tactic_state=state.tactic_state,
            comment=comment,
            depth=state.depth + 1,
            header=state.header,
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
        return hash(state.code())
