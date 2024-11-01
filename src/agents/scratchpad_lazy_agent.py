from typing import Any, Optional

import numpy as np
from wayfinder.games import *

from src.agents.scratchpad_lazy_agent_prompts import *
from src.lean.scratchpad_lean_game import LeanGame, LeanMove, LeanState
from src.workers.worker import *


class ScratchpadLazyAgent(Agent[LeanGame, LeanState, LeanMove]):
    def __init__(self,
                 game: LeanGame,
                 worker: Worker,
                 request_formula: str,
                 max_num_completions: int,
                 valueless: bool = False,
                 ):
        super().__init__(game=game)
        self.worker = worker
        self.request_formula = request_formula
        self.max_num_completions = max_num_completions
        self.valueless = valueless

    async def max_moves(self, state: LeanState) -> int:
        return self.max_num_completions

    async def require_new_move(
        self,
        state: LeanState,
        min_num_moves: int,
        max_num_moves: Optional[int] = None
    ) -> bool:
        """Generate new moves by first updating scratchpad, then generating new code."""
        if max_num_moves is None:
            max_num_moves = min_num_moves

        if max_num_moves > self.max_num_completions:
            raise ValueError(
                "The number of completions requested is greater than the maximum number of completions.")

        if state not in self.active_move_cache:
            self.active_move_cache[state] = []

        while True:
            num_queries_needed = max_num_moves - \
                len(self.active_move_cache[state])
            if num_queries_needed <= 0:
                break

            # First get scratchpad additions
            scratchpad_completions = await self.generate_scratchpad(
                state,
                num_queries_needed,
                max_num_moves ** 0.1
            )

            # Then get code completions for each scratchpad addition
            active_move_set = set(self.active_move_cache[state])
            for scratchpad_completion in scratchpad_completions:
                code_completions = await self.generate_code(
                    state,
                    scratchpad_completion['text'],
                    1,  # One code completion per scratchpad addition
                    max_num_moves ** 0.1
                )

                code_completion = code_completions[0]

                move = LeanMove(
                    new_code=code_completion['text'],
                    scratchpad_append=scratchpad_completion['text']
                )
                # Combined log probability
                probability = np.exp(
                    scratchpad_completion['cumulative_logprob'] +
                    code_completion['cumulative_logprob']
                )

                if move not in active_move_set:
                    self.active_move_cache[state].append(move)
                    self.policy_cache[hash((state, move))] = probability
                    active_move_set.add(move)

            if len(self.active_move_cache[state]) >= min_num_moves:
                break

        return True

    async def generate_scratchpad(self, state: LeanState, n: int, temperature: float) -> list[dict]:
        """Generate scratchpad additions using chat completion."""
        messages = get_scratchpad_messages(
            state.annotated_code,
            state.scratchpad
        )

        completion = await self.worker.query(
            task={
                'messages': messages,
                'n': n,
                'temperature': temperature,
                'channel': self.worker.name,
            },
            channel='chat'
        )
        return completion['result']

    async def generate_code(self, state: LeanState, new_scratchpad: str, n: int, temperature: float) -> list[dict]:
        """Generate new code using chat completion."""
        messages = get_code_messages(
            state.annotated_code,
            state.scratchpad + new_scratchpad
        )

        completion = await self.worker.query(
            task={
                'messages': messages,
                'n': n,
                'temperature': temperature,
                'channel': self.worker.name,
            },
            channel='chat'
        )

        result = completion['result']
        for r in result:
            if "```" in r['text']:
                r['text'] = r['text'][:r['text'].find("```")]

        return result

    async def value(self, state: LeanState) -> float:
        """Estimate value using chat completion."""
        if self.valueless:
            return 0

        messages = get_value_messages(
            state.annotated_code,
            state.scratchpad
        )

        value = await self.worker.query(
            task={
                'messages': messages,
                'channel': self.worker.name,
            },
            channel='chat'
        )

        return parse_value_response(value['result'][0]['text'], self.worker.logger)['rating']

    async def policy(self, state: LeanState, move: LeanMove) -> float:
        if hash((state, move)) in self.policy_cache:
            return self.policy_cache[hash((state, move))]
        raise ValueError("Move probability not calculated.")


def parse_value_response(output: str, logger: logging.Logger) -> dict:
    """Parse value response, returning normalized rating between -1 and 1."""
    # Find content inside \boxed{}
    start = output.find('\\boxed{')
    if start == -1:
        logger.warning(
            f"No \\boxed{{}} found. Returning 0. Output: {output}")
        return {'rating': 0}

    start += len('\\boxed{')
    end = output.find('}', start)
    if end == -1:
        logger.warning(
            f"Unclosed \\boxed{{}}. Returning 0. Output: {output}")
        return {'rating': 0}

    boxed_content = output[start:end].strip()
    # Extract just the numbers and decimal points
    rating_str = ''.join(
        c for c in boxed_content if c.isdigit() or c == '.')
    try:
        rating = float(rating_str)
        return {'rating': (rating - 50) / 50}
    except ValueError:
        logger.warning(f"Rating not a number. Returning 0. Output: {output}")
        return {'rating': 0}
