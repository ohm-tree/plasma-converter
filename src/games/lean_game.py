import os
import random
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple

import numpy as np

from src.games.concurrent import handler, on_startup, require_ready
from src.games.game import ConcurrentGameState, ConcurrentMetaGameState
from src.games.lean_game_core import LeanGameMove, LeanGameState, LeanGameStateError
from src.uct.uct_node import UCTNode
from src.workers.types import CompletionTaskType, LeanTaskType, PolicyValueTaskType
from src.workers.worker import (
    TaskIdentifier,
    TaskType,
    WorkerIdentifer,
    WorkerResponse,
    WorkerTask,
)

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


@dataclass(frozen=True)
class MetaLeanGameMove:
    """
    A move in the Lean game.
    """
    comment: str


class MetaLeanGameState(ConcurrentMetaGameState[LeanGameState, MetaLeanGameMove]):
    def __init__(self,
                 worker_id: WorkerIdentifer,
                 internal_state: LeanGameState,
                 comment: str,
                 gen_comments: Optional[List[str]] = None,
                 ):
        """
        """
        super().__init__(worker_id=worker_id,
                         internal_state=internal_state)

        self.comment = comment
        self.gen_comments = gen_comments or []

        # We don't want to re-generate a child when we re-do an action,
        # so pointers to the MetaLeanGameState children are stored here.
        self.children = {}

        # This is just a unique identifier for the state.
        self._id = self.state._id

        self._policy = None
        self._value = None

    def human_json(self) -> dict:
        """
        Return a human-readable JSON representation of the state.
        """
        return {
            "id": self._id,
            "comment": self.comment,
            "gen_comments": self.gen_comments,
            "state": self.state.human_json()
        }

    @classmethod
    def saves(cls, states: List['MetaLeanGameState'], filename: str):
        """
        Save a collection of states to a file.

        This is super important. During MCTS,
        this is what gets saved to the replay buffer.
        """
        np.save(filename, [state.human_json() for state in states])

    @classmethod
    def loads(cls, filename: str) -> List['MetaLeanGameState']:
        """
        Load a collection of states from a file.
        """
        states = np.load(filename, allow_pickle=True)
        return [cls(**state) for state in states]

    @classmethod
    def starting_state(cls,
                       worker_id: WorkerIdentifer,
                       problem: str,
                       tactic_state: str,
                       header: str = LEAN4_DEFAULT_HEADER,
                       max_depth: int = 20) -> 'MetaLeanGameState':

        internal_state = LeanGameState.starting_state(
            worker_id=worker_id,
            problem=problem,
            header=header,
            tactic_state=tactic_state,
            max_depth=max_depth
        )

        return MetaLeanGameState(
            worker_id=worker_id,
            internal_state=internal_state,
            comment="",
            gen_comments=[]
        )

    def next_state(self, action: MetaLeanGameMove) -> LeanGameState:
        """
        Returns the next state of the game given a current state and action.
        Requires that the state is non-terminal and action is legal.
        """
        if action not in self.active_moves():
            raise LeanGameStateError(
                f"Action {action} not in active moves {self.active_moves()}")
        if action in self.children:
            return self.children[action]
        new_state = self.state.next_state(action)
        self.children[action] = MetaLeanGameState(
            worker_id=self.worker_id,
            internal_state=new_state,
            comment="",
            gen_comments=[]
        )
        return self.children[action]

    def make_root(self) -> None:
        self.state.make_root()

    @require_ready
    def policy(self) -> np.ndarray:
        return self._policy

    @require_ready
    def value(self) -> float:
        return self._value

    @require_ready
    def get_active_move(self, index: int) -> MetaLeanGameMove:
        return self.gen_comments[index]

    @require_ready
    def active_moves(self) -> List[MetaLeanGameMove]:
        return self.gen_comments

    @require_ready
    def index_active_move(self, move: MetaLeanGameMove) -> int:
        return self.gen_comments.index(move)

    @require_ready
    def len_active_moves(self) -> int:
        return len(self.gen_comments)

    @require_ready
    def require_new_move(self) -> None:
        raise NotImplementedError(
            "This function is not implemented for LeanGameState")

    def __hash__(self) -> int:
        """
        Hash the state based on the id.
        """
        return self._id

    def human_printout(self) -> str:
        res = ""

        def fancy_field(name: str, value: str, length=80, tick='-') -> str:
            if type(name) is not str:
                name = str(name)
            if type(value) is not str:
                value = str(value)

            res = name.center(length, tick) + "\n"
            res += value
            if len(value) == 0:
                res += "[Empty Field]\n"
            elif value[-1] != '\n':
                res += '\n'
                res += "[Missing newline]\n"
            return res

        res += fancy_field("Header", self.state.header)
        res += fancy_field("Problem", self.state.problem)
        res += fancy_field("Old Code", self.state.old_code)
        res += fancy_field("Comment", self.comment)
        res += fancy_field("Completed Rollout without Truncation",
                           self.state.new_code)
        res += fancy_field("Valid Truncation of New Code",
                           self.state.valid_code)
        res += fancy_field("Generated Comments",
                           '\n'.join(i.comment for i in self.gen_comments))

        res += fancy_field("Old Tactic State", self.state.old_tactic_state)
        res += fancy_field("New Tactic State", self.state.tactic_state)

        res += fancy_field("Meta", f"Win: {self.state._win}, Dead: {self.state._dead}\n"
                           f"Depth: {self.state.depth} Number of Children: {len(self.children)}\n")
        return res

    @ on_startup
    def pre_LLM_rollout(self) -> Iterator[WorkerTask]:
        """
        This function is called before the LLM rollout is done.
        It generates a prompt for the LLM.
        """

        prompt = 'Complete the following Lean 4 code.\nHere is a hint:\n' + self.comment.strip() + \
            '\nThe tactic state is:\n' + \
            self.state.old_tactic_state.strip()+'\n```lean\n' + self.state.header + self.state.problem + \
            self.state.old_code

        yield WorkerTask(
            head_id=self.worker_id,
            task_id=TaskIdentifier(
                task_type=CompletionTaskType,
                task_idx=hash(self)
            ),
            task=prompt,
        )

    @handler(CompletionTaskType)
    def post_LLM_rollout(self, completion: WorkerResponse) -> Iterator[WorkerTask]:
        """
        This function is called after the LLM rollout is done.
        """
        new_code: str = completion.response
        if new_code.endswith('```'):
            new_code = new_code[:-3]
        if not new_code.endswith('\n'):
            new_code += '\n'
        self.state.new_code = new_code

        return self.state.startup(callback=self.pre_comments)

    def pre_comments(self) -> Iterator[WorkerTask]:
        """
        This function is called before the comments are generated.
        """
        # print("#" * 80)
        # print(self.state.header, self.state.problem,
        #       self.state.old_code, self.state.tactic_state)
        task = {
            "header": self.state.header,
            "problem": self.state.problem,
            "old_code": self.state.old_code,
            "tactic_state": self.state.tactic_state
        }

        yield WorkerTask(
            head_id=self.worker_id,
            task_id=TaskIdentifier(
                task_type=PolicyValueTaskType,
                task_idx=hash(self)
            ),
            task=task,
        )

    @handler(PolicyValueTaskType)
    def post_comments(self, results: WorkerResponse) -> None:
        """
        This function is called after the comments are generated.
        """
        # print(results)
        # print(type(results))
        # print(results.__dict__)
        # print(results.response)
        self.gen_comments = tuple(
            MetaLeanGameMove(comment) for comment in results.response['comments']
        )
        self._ready = True
        self._policy = np.array(results.response['policy'])
        self._value = results.response['value']
        self.finish()

    @handler(LeanTaskType)
    def post_process(self, lean_output: WorkerResponse) -> Iterator[WorkerTask]:
        """
        This function is called after the Lean task is done.
        """
        yield from self.state.post_process(lean_output)
