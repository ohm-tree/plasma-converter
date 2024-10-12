import os
import random
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple

import numpy as np

from src.games.concurrent import finisher, handler, on_startup, require_ready
from src.games.game import ConcurrentGameState, ConcurrentMetaGameState
from src.games.lean_game_core import LeanGameMove, LeanGameState, LeanGameStateError
from src.uct.uct_node import UCTNode
from src.workers.types import CompletionTaskType, LeanTaskType, PolicyValueTaskType, PolicyValuePostProcessTaskType
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
    code: str
    # comment: str


class MetaLeanGameState(ConcurrentMetaGameState[LeanGameState, MetaLeanGameMove]):
    def __init__(self,
                 worker_id: WorkerIdentifer,
                 internal_state: LeanGameState,
                 code: str,
                 next_moves: Optional[List[str]] = None,
                 ):
        """
        """
        super().__init__(worker_id=worker_id,
                         internal_state=internal_state)

        self.code = code
        self.next_moves = next_moves or []

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
            "code": self.code,
            "next_moves": self.next_moves,
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
                       max_depth: int,
                       header: str = LEAN4_DEFAULT_HEADER) -> 'MetaLeanGameState':

        internal_state = LeanGameState.starting_state(
            worker_id=worker_id,
            problem=problem,
            header=header,
            # old_tactic_state=tactic_state,
            tactic_state=tactic_state,
            max_depth=max_depth
        )

        return MetaLeanGameState(
            worker_id=worker_id,
            internal_state=internal_state,
            code="",
            next_moves=[]
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
            code="",
            next_moves=[]
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
        return self.next_moves[index]

    @require_ready
    def active_moves(self) -> List[MetaLeanGameMove]:
        return self.next_moves

    @require_ready
    def index_active_move(self, move: MetaLeanGameMove) -> int:
        return self.next_moves.index(move)

    @require_ready
    def len_active_moves(self) -> int:
        return len(self.next_moves)

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
        res += fancy_field("New code", self.code)
        res += fancy_field("Completed Rollout without Truncation",
                           self.state.new_code)
        res += fancy_field("Valid Truncation of New Code",
                           self.state.valid_code)
        res += fancy_field("Generated Moves",self.next_moves)

        res += fancy_field("Old Tactic State", self.state.old_tactic_state)
        res += fancy_field("New Tactic State", self.state.tactic_state)

        res += fancy_field("Meta", f"Win: {self.state._win}, Dead: {self.state._dead}\n"
                           f"Depth: {self.state.depth} Number of Children: {len(self.children)}\n")
        return res

    @on_startup
    def pre_LLM_rollout(self) -> Iterator[WorkerTask]:
        """
        This function is called before the LLM rollout is done.
        It generates a prompt for the LLM.
        """

        prompt = 'Complete the following Lean 4 code with explanatory comments.' + \
            '```lean\n' + self.state.header + self.state.problem + \
            self.state.old_code + \
            "\n  /--\n" + self.state.old_tactic_state.strip() + "\n-/" + \
            "  --" 

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
        
        self.state.new_code = new_code
        print("new_code", new_code)

        return self.state.startup(callback=self.pre_query)

    def pre_query(self) -> Iterator[WorkerTask]:
        """
        This function is called before the moves are generated.
        """

        # if we're terminal, we don't need to do any of this.
        if self.state.terminal():
            yield from self.finish()
            return
        # print("#" * 80)
        # print(self.state.header, self.state.problem,
        #       self.state.old_code, self.state.tactic_state)
        # print(self.state.old_tactic_state)
        task = {
            "header": self.state.header,
            "problem": self.state.problem,
            "old_code": self.state.old_code,
            "tactic_state": self.state.tactic_state,
            "old_tactic_state": self.state.old_tactic_state
        }

        yield WorkerTask(
            head_id=self.worker_id,
            task_id=TaskIdentifier(
                task_type=PolicyValueTaskType,
                task_idx=hash(self)
            ),
            task=task
        )

    @handler(PolicyValueTaskType)
    @finisher
    def post_query(self, results: WorkerResponse) -> Iterator[WorkerTask]:
        """
        This function is called after the code snippets (comments + completion) are generated
        """
        # print(results)
        # print(type(results))
        # print(results.__dict__)
        # print(results.response)
        self.next_moves = [
            MetaLeanGameMove(move) for move in results.response['moves']
        ]
        self._policy = np.array(results.response['policy'])
        self._value = results.response['value']

        yield from ()
