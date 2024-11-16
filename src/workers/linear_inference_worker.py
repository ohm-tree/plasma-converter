"""
This file contains the main entrypoint for linear inference.
It is responsible for running the linear inference algorithm
on a given problem, saving the results to disk, and logging
the results.
"""

import multiprocessing
from multiprocessing import sharedctypes
from typing import Union

from src.agents.lazy_agent import LazyLeanAgent
from src.agents.scratchpad_lazy_agent import ScratchpadLazyAgent
from src.lean.lean_game import LeanGame, LeanState
from src.lean.scratchpad_lean_game import ScratchpadGame, ScratchpadState
from src.workers.inference_worker import InferenceWorker


class LinearWorker(InferenceWorker):
    def __init__(self,
                 global_config: dict,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: dict[str, multiprocessing.Queue],
                 values: dict[str, sharedctypes.Synchronized],
                 **kwargs  # Unused
                 ):
        super().__init__(
            name="linear" + "_" + str(task_id),
            worker_type="linear",
            worker_idx=task_id,
            global_config=global_config,
            config=config,
            queues=queues,
            values=values,
            run_name=run_name,
        )

        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )

    async def solve(self, game: Union[LeanGame, ScratchpadGame]) -> dict:
        if isinstance(game, LeanGame):
            return await self.solve_lean(game)
        elif isinstance(game, ScratchpadGame):
            return await self.solve_scratchpad(game)

    async def solve_lean(self, game: LeanGame) -> dict:
        agent: LazyLeanAgent = LazyLeanAgent(
            game=game,
            worker=self,
            # We don't care about this value, amount_to_request is never called.
            request_formula=None,
            # For linear inference, we only need one completion
            max_num_completions=1
        )

        state: LeanState = await game.starting_state()

        states: list[LeanState] = []

        states.append(state)

        while not await game.terminal(state):
            self.logger.info(state.__str__())
            # Obtains the next move from the agent
            res = await agent.require_new_move(state, 1, 1)
            if not res:
                # We failed to find a new move!
                self.logger.error("Failed to find a new move!")
                break
            # 0 is the index of the move
            action = await agent.get_active_move(state, 0)
            state = await game.next_state(state, action)
            states.append(state)

        if (await game.terminal(state)):
            reward = await game.reward(state)
        else:
            reward = game.death_value

        self.logger.info(state.__str__())

        return {
            'states': states,
            "result": reward
        }

    async def solve_scratchpad(self, game: ScratchpadGame) -> dict:
        agent: ScratchpadLazyAgent = ScratchpadLazyAgent(
            game=game,
            worker=self,
            # We don't care about this value, amount_to_request is never called.
            request_formula=None,
            # For linear inference, we only need one completion
            max_num_completions=1
        )

        state: ScratchpadState = await game.starting_state()

        states: list[ScratchpadState] = []
        states.append(state)

        while not await game.terminal(state):
            self.logger.info(state.__str__())

            res = await agent.require_new_move(state, 1, 1)
            if not res:
                # We failed to find a new move!
                self.logger.error("Failed to find a new move!")
                break

            action = await agent.get_active_move(state, 0)
            state = await game.next_state(state, action)
            states.append(state)

        if (await game.terminal(state)):
            reward = await game.reward(state)
        else:
            reward = game.death_value

        self.logger.info(state.__str__())

        return {
            'states': states,
            "result": reward
        }
