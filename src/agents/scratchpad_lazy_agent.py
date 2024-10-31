from typing import Any, Optional

import numpy as np
from wayfinder.games import *

from src.lean.scratchpad_lean_game import LeanGame, LeanMove, LeanState
from src.workers.worker import *

SCRATCHPAD_PROMPT = """Here is the current state of a Lean proof:
```lean4
{annotated_code}
```
Here are my current notes in the scratchpad:
{scratchpad}

Please analyze the current state of the proof, especially noting the error messages.
What should we try next? What insights can we draw from our previous attempts?
Write your thoughts as a continuation of the scratchpad.

Your response should start immediately without any preamble.
"""

CODE_PROMPT = """Here is the current state of a Lean proof, with error annotations:
```lean4
{annotated_code}
```

Here are my notes about the proof:
{scratchpad}

Please rewrite the proof, taking into account the error messages and insights from the notes.
"""

VALUE_PROMPT = """Here is a Lean proof with error annotations:
```lean4
{annotated_code}
```

Here are the notes from attempting this proof:
{scratchpad}

Based on the state of this code and the included annotations, rate the likelihood that this proof will succeed on a scale of 0 (very unlikely) to 100 (very likely).
Please reason step by step about what's working and what isn't, then put your final answer in \\boxed{{}}
"""


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

                for code_completion in code_completions:
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
        """Generate scratchpad additions."""
        prompt = SCRATCHPAD_PROMPT.format(
            annotated_code=state.annotated_code.to_annotated_string(),
            scratchpad=state.scratchpad
        )

        completion = await self.worker.query(
            task={
                'prompt': prompt,
                'n': n,
                'temperature': temperature,
                'channel': self.worker.name,
            },
            channel='completion'
        )
        return completion['result']

    async def generate_code(self, state: LeanState, new_scratchpad: str, n: int, temperature: float) -> list[dict]:
        """Generate new code given the state and new scratchpad addition."""
        prompt = CODE_PROMPT.format(
            annotated_code=state.annotated_code.to_annotated_string(),
            scratchpad=state.scratchpad + new_scratchpad
        )

        completion = await self.worker.query(
            task={
                'prompt': prompt,
                'n': n,
                'temperature': temperature,
                'channel': self.worker.name,
            },
            channel='completion'
        )

        result = completion['result']
        for r in result:
            if r['text'].endswith('```'):
                r['text'] = r['text'][:-3]
            if not r['text'].endswith('\n'):
                r['text'] += '\n'
        return result

    async def value(self, state: LeanState) -> float:
        """Estimate value based on annotated code and scratchpad."""
        if self.valueless:
            return 0

        prompt = VALUE_PROMPT.format(
            annotated_code=state.annotated_code.to_annotated_string(),
            scratchpad=state.scratchpad
        )

        value = await self.worker.query(
            task={
                'prompt': prompt,
                'channel': self.worker.name,
            },
            channel='value'
        )

        return parse_value_response(value['result'][0]['text'], self.worker.logger)['rating']

    async def policy(self, state: LeanState, move: LeanMove) -> float:
        if hash((state, move)) in self.policy_cache:
            return self.policy_cache[hash((state, move))]
        raise ValueError("Move probability not calculated.")


def parse_value_response(output: str, logger: logging.Logger) -> dict:
    """Parse value response, returning normalized rating between -1 and 1."""
    output_end = output.find('}')
    if output_end != -1:
        output = output[:output_end].strip()
    output = ''.join(c for c in output if c.isdigit() or c == '.')
    try:
        rating = float(output)
    except ValueError:
        logger.warning(f"Rating not a number. Returning 0. Output: {output}")
        return {'rating': 0}
    return {'rating': (rating - 50) / 50}
