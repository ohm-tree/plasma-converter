from typing import Any, Optional

import numpy as np
from wayfinder.games import *

from src.agents.scratchpad_lazy_agent_prompts import (
    code_prompter,
    scratchpad_prompter,
    value_prompter,
)
from src.lean.scratchpad_lean_game import (
    ScratchpadGame,
    ScratchpadMove,
    ScratchpadState,
)
from src.workers.worker import *


class ScratchpadLazyAgent(Agent[ScratchpadGame, ScratchpadState, ScratchpadMove]):
    def __init__(self,
                 game: ScratchpadGame,
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

    async def max_moves(self, state: ScratchpadState) -> int:
        return self.max_num_completions

    async def amount_to_request(self,
                                state: ScratchpadState,
                                current_num_children: int,
                                num_visits: int,
                                child_num_visits: np.ndarray,
                                child_priors: np.ndarray,
                                child_total_value: np.ndarray,
                                child_Q: np.ndarray,
                                child_U: np.ndarray,
                                c: float,
                                expand_initial_value: float) -> tuple[int, Optional[int]]:
        """
        Returns the amount of moves to request.
        """
        if self.request_formula == "sqrt":
            max_moves = await self.max_moves(state)
            current = await self.len_active_moves(state)
            if current < 4:
                return (max(1, current), min(4, max_moves))
            if current * current < num_visits:
                return (min(current, max_moves), min(current * 2, max_moves))
            # We don't need more
            return (min(current, max_moves), None)
        if self.request_formula == "log":
            max_moves = await self.max_moves(state)
            current = await self.len_active_moves(state)
            if current < 4:
                return (max(1, current), min(4, max_moves))
            if 2 ** current < num_visits:
                return (min(current, max_moves), min(current * 2, max_moves))
            return (min(current, max_moves), None)
        if self.request_formula == "conservative":
            max_moves = await self.max_moves(state)
            current = await self.len_active_moves(state)
            if current < 4:
                return (max(1, current), min(4, max_moves))

            current_best = np.max(child_Q + c * child_U)
            if expand_initial_value >= current_best:
                return (min(current, max_moves), min(current * 2, max_moves))
            return (min(current, max_moves), None)
        if self.request_formula == "pure":
            max_moves = await self.max_moves(state)
            current = await self.len_active_moves(state)
            if current < 4:
                return (max(1, current), min(4, max_moves))

            current_best = np.max(child_Q + c * child_U)

            # simulate a maximally likely new child.
            remaining_probability = max(1 - np.sum(child_priors), 0)
            new_U = remaining_probability * np.sqrt(num_visits)

            if expand_initial_value + c * new_U >= current_best:
                return (min(current, max_moves), min(current * 2, max_moves))
            return (min(current, max_moves), None)

        raise ValueError(f"Unknown request formula: {self.request_formula}")

    async def require_new_move(
        self,
        state: ScratchpadState,
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

                move = ScratchpadMove(
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

    async def generate_scratchpad(self, state: ScratchpadState, n: int, temperature: float) -> list[dict]:
        """Generate scratchpad additions using chat completion."""
        messages = scratchpad_prompter.create_prompt(
            problem_statement=self.game.problem,
            natural_language_proof=self.game.natural_language_proof,
            annotated_code=state.annotated_code,
            scratchpad=state.scratchpad
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

    async def generate_code(self, state: ScratchpadState, new_scratchpad: str, n: int, temperature: float) -> list[dict]:
        """Generate new code using chat completion."""
        messages = code_prompter.create_prompt(
            natural_language_proof=self.game.natural_language_proof,
            annotated_code=state.annotated_code,
            problem_statement=self.game.problem,
            scratchpad=state.scratchpad + new_scratchpad
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

    async def value(self, state: ScratchpadState) -> float:
        """Estimate value using chat completion."""
        if self.valueless:
            return 0

        messages = value_prompter.create_prompt(
            problem_statement=self.game.problem,
            natural_language_proof=self.game.natural_language_proof,
            scratchpad=state.scratchpad,
            annotated_code=state.annotated_code,
        )

        value = await self.worker.query(
            task={
                'messages': messages,
                'channel': self.worker.name,
            },
            channel='chat'
        )

        return parse_value_response(value['result'][0]['text'], self.worker.logger)['rating']

    async def policy(self, state: ScratchpadState, move: ScratchpadMove) -> float:
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
