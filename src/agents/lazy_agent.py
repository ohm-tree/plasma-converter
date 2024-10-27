from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import numpy as np
from wayfinder.games import *

from src.lean.lean_game import LeanGame, LeanMove, LeanState
from src.workers.worker import *

# Todo: prompts are *data*; they can be stored in files and read from configs.

COMPLETION_PROMPT = """Complete the following Lean 4 code.
The tactic state is:
```lean4
{tactic_state}
```
Here is the Lean 4 code so far:
```lean4
{header}
{problem}
{old_code}
"""  # No 3 backticks at the end, so the model can complete the code.


CONTEXT_PROMPT = """This is a partial Lean 4 proof.
```lean4
{header}
{problem}
{old_code}
```
Here is the tactic state at this point:
```lean4
{tactic_state}
```
QUESTION:
Please summarize what we have proven so far.
Please summarize the current tactic state of the proof.
Then, please discuss whether or not the proof is on the right track. Are we proving useful lemmas? Are we using the right tactics? Are we missing any key insights?
ANSWER:
"""

VALUE_PROMPT = """The current state of the proof:
{context}
Here is the tactic state at this point:
```lean4
{tactic_state}
```
Rate the likelihood that this proof will succeed on a scale of 0 (very unlikely) to 100 (very likely).
Please reason step by step, and put your final answer within \\boxed{{}}.
"""


def parse_value_response(output: str, logger: logging.Logger) -> dict:
    """
    Parse the output of the policy-value worker into a dict.
    """
    output_start = output.find(r'\boxed{')
    output_end = output.find('}', output_start)
    if output_start == -1 or output_end == -1:
        logger.warning(
            f"Output does not contain a boxed value. Output: {output}")
        return {
            'rating': 0
        }

    try:
        rating = float(output)
    except ValueError:
        logger.warning(
            f"Rating not a number. Returning 0. Output: {output}")
        return {
            'rating': 0
        }

    return {
        'rating': (rating - 50) / 50
    }


class LazyLeanAgent(Agent[LeanGame, LeanState, LeanMove]):
    def __init__(self,
                 game: LeanGame,
                 worker: Worker,
                 request_formula: str,
                 max_num_completions: int,
                 valueless: bool = False,
                 ):
        """
        """
        super().__init__(game=game)
        self.worker = worker
        self.request_formula = request_formula
        self.max_num_completions = max_num_completions

        self.valueless = valueless

    async def max_moves(self, state: LeanState) -> int:
        """
        Returns the maximum number of moves.
        """
        return self.max_num_completions

    async def amount_to_request(self,
                                state: LeanState,
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
            for i in range(len(completions)):
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

        prompt = COMPLETION_PROMPT.format(
            tactic_state=state.tactic_state,
            header=self.game.header,
            problem=self.game.problem,
            old_code=state.code,
        )

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

        if len(res) < num_completions:
            self.worker.logger.warning(
                f"Only received {len(res)} completions, expected {num_completions}.")
            self.worker.logger.warning(
                "This is likely due to the LLM query being too long.")
            self.worker.logger.warning(
                "The expected output on failure is [{'text': '', 'token_ids': (), 'cumulative_logprob': 0.0}]")
            self.worker.logger.warning(f"Length of Prompt: {len(prompt)}")
            self.worker.logger.warning(f"Response: {res}")

        for i in range(len(res)):
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
        if self.valueless:
            return 0

        lean_game_dict = {
            "header": self.game.header,
            "problem": self.game.problem,
            "old_code": state.code,
            "tactic_state": state.tactic_state,
            'channel': self.worker.name,
        }

        context = await self.worker.query(
            task={
                'prompt': CONTEXT_PROMPT.format(**lean_game_dict),
                'channel': self.worker.name,
            },
            channel='context'
        )

        value = await self.worker.query(
            task={
                'prompt': VALUE_PROMPT.format(**lean_game_dict, context=context['result']),
                'channel': self.worker.name,
            },
            channel='value'
        )

        return parse_value_response(
            value['result'],
            self.worker.logger
        )
