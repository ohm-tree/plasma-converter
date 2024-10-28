from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import numpy as np
from wayfinder.games import *

from src.lean.lean_game import LeanGame, LeanMove, LeanState
from src.workers.worker import *

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


class LazyLeanAgent(Agent[LeanGame, LeanState, LeanMove]):
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

        On the first pass, attempts to generate exactly max_num_moves completions.

        Then, continues generating completions until there are at least min_num_moves completions.
        """

        if max_num_moves is None:
            max_num_moves = min_num_moves

        if max_num_moves > self.max_num_completions:
            raise ValueError(
                "The number of completions requested is greater than the maximum number of completions.")

        if state not in self.active_move_cache:
            self.active_move_cache[state] = []

        # TODO: as the number of queries increases, we should scale the temperature up for more variety.
        while len(self.active_move_cache[state]) < min_num_moves:
            num_queries_needed = max_num_moves - \
                len(self.active_move_cache[state])

            completions = await self.LLM_rollout(state, num_queries_needed, 1.0)

            active_move_set = set(self.active_move_cache[state])
            for i in range(num_queries_needed):
                move = LeanMove(completions[i]['text'])
                probability = completions[i]['cumulative_logprob']
                probability = np.exp(probability)
                if move not in active_move_set:
                    self.active_move_cache[state].append(move)
                    self.policy_cache[(state, move)] = probability
                    active_move_set.add(move)

        return True

    async def LLM_rollout(self, state: LeanState, num_completions: int, temperature: float) -> str:
        """
        Completes a state.
        """
        prompt = 'Complete the following Lean 4 code with explanatory comments.' + \
            '```lean\n' + self.game.header + self.game.problem + \
            state.code + \
            "\n  /-- Tactic state:\n" + '\n'.join(['  ' + line for line in state.tactic_state.strip().splitlines()]) + "\n  -/\n" + \
            "  --" 
        # prompt = 'Complete the following Lean 4 code.\n' + \
        #     'The tactic state is:\n' + \
        #     state.tactic_state.strip()+'\n```lean\n' + self.game.header + self.game.problem + \
        #     state.code

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

        def context_prompt(lean_game_dict: dict) -> str:
            """
            Generate a prompt for the policy-value worker to suggest comments.
            """
            res = """This is a partial Lean 4 proof.
        ```lean4
        """
            res += lean_game_dict['header'] + \
                lean_game_dict['problem'] + lean_game_dict['old_code']
            res += """
        ```
        Here is the tactic state at this point:
        ```lean4
        """
            res += lean_game_dict['tactic_state']
            res += f"""
        ```
        QUESTION:
        Please summarize what we have proven so far.
        Please summarize the current tactic state of the proof.
        Then, please discuss whether or not the proof is on the right track. Are we proving useful lemmas? Are we using the right tactics? Are we missing any key insights?

        ANSWER:
        """
            return res

        def value_prompt(lean_game_dict: dict, discussion_context: str) -> str:
            # We should call the LLM now.

            res = discussion_context + f"""Here is the tactic state at this point:
        ```lean4
        """
            res += lean_game_dict['tactic_state']
            res += f"""
        ```
        Rate the likelihood that this proof will succeed on a scale of 0 (very unlikely) to 100 (very likely).
        Please delimit this rating with a single number inside <RATING></RATING> tags.

        <RATING>"""
            return res

        def parse_value_response(output: str, logger: logging.Logger) -> dict:
            """
            Parse the output of the policy-value worker into a dict.
            """
            if "</RATING>" not in output:
                logger.warning("Rating not found in output. Returning 0.")
                return {
                    'rating': 0
                }

            output = output[:output.find('</RATING>')]

            try:
                rating = float(output)
            except ValueError:
                logger.warning(
                    f"Rating not a number. Returning 0. Output: {output}")
                return {
                    'rating': 0
                }

            return {
                'rating': (rating - 50) / 100
            }
        # print("#" * 80)
        # print(self.state.header, self.state.problem,
        #       self.state.old_code, self.state.tactic_state)
        lean_game_dict = {
            "header": self.game.header,
            "problem": self.game.problem,
            "old_code": state.code,
            "tactic_state": state.tactic_state,
            'channel': self.worker.name,
        }

        context = await self.worker.query(
            task={
                'prompt': context_prompt(lean_game_dict),
                'channel': self.worker.name,
            },
            channel='context'
        )

        value = await self.worker.query(
            task={
                'prompt': value_prompt(lean_game_dict, context),
                'channel': self.worker.name,
            },
            channel='value'
        )

        return parse_value_response(value)
