from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import numpy as np
from wayfinder.games import *

from src.lean.lean_game import LeanGame, LeanMove, LeanState
from src.workers.worker import *

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


class LazyValuelessLeanAgent(Agent[LeanGame, LeanState, LeanMove]):
    def __init__(self,
                 game: LeanGame,
                 worker: Worker,
                 max_num_completions: int = 100,
                 ):
        """
        """
        super().__init__(game=game)
        self.worker = worker

        self.max_num_completions = max_num_completions

    async def max_moves(self, state: LeanState) -> int:
        """
        Returns the maximum number of moves.
        """
        return self.max_num_completions

    async def require_new_move(
        self,
        state: LeanState,
        min_num_moves: int,
        max_num_moves: Optional[int] = None
    ) -> bool:
        """
        Behavior:

        On each pass, attempts to generate exactly max_num_moves completions.

        Always makes at least one pass, and stops when there are at least min_num_moves completions.
        """

        if max_num_moves is None:
            max_num_moves = min_num_moves

        if max_num_moves > self.max_num_completions:
            raise ValueError(
                "The number of completions requested is greater than the maximum number of completions.")

        if state not in self.active_move_cache:
            self.active_move_cache[state] = []

        # TODO: as the number of queries increases, we should scale the temperature up for more variety.
        while True:
            num_queries_needed = max_num_moves - \
                len(self.active_move_cache[state])

            if num_queries_needed <= 0:
                # Just make it bullet-proof.
                break

            # Scale the temperature up as the number of queries increases.
            completions = await self.LLM_rollout(state, num_queries_needed, max_num_moves ** 0.1)

            active_move_set = set(self.active_move_cache[state])
            for i in range(num_queries_needed):
                move = LeanMove(completions[i]['text'])
                probability = completions[i]['cumulative_logprob']
                probability = np.exp(probability)
                if move not in active_move_set:
                    self.active_move_cache[state].append(move)
                    self.policy_cache[hash((state, move))] = probability
                    active_move_set.add(move)

            if len(self.active_move_cache[state]) >= min_num_moves:
                break

        return True

    async def LLM_rollout(self, state: LeanState, num_completions: int, temperature: float) -> str:
        """
        Completes a state.
        """

        # print("Attempting to get completions")

        prompt = 'Complete the following Lean 4 code.\n' + \
            'The tactic state is:\n' + \
            state.tactic_state.strip()+'\n```lean\n' + self.game.header + self.game.problem + \
            state.code

        completion = await self.worker.query(
            task={
                'prompt': prompt,
                'n': num_completions,
                'temperature': temperature,
                'channel': self.worker.name,
            },
            channel='completion'
        )

        # TODO: make this a named dict
        res: list[dict[str, Any]] = completion['result']

        for i in range(num_completions):
            if res[i]['text'].endswith('```'):
                res[i]['text'] = res[i]['text'][:-3]
            if not res[i]['text'].endswith('\n'):
                res[i]['text'] += '\n'
        return res

    async def policy(self, state: LeanState, move: LeanMove) -> float:
        """
        Returns the policy for the game state.
        """
        if hash((state, move)) in self.policy_cache:
            return self.policy_cache[hash((state, move))]
        else:
            raise ValueError(
                "The probability of this move in this state has not been calculated.")

    async def value(self, state: LeanState):
        """
        This function is called before the comments are generated.
        """

        return 0.0
