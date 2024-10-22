"""
This file contains the main entrypoint for linear inference.
It is responsible for running the linear inference algorithm
on a given problem, saving the results to disk, and logging
the results.
"""

import multiprocessing

from src.agents.lazy_agent import LazyLeanAgent
from src.lean.lean_game import LeanGame, LeanState
from src.workers.inference_worker import InferenceWorker


class LinearWorker(InferenceWorker):
    def __init__(self,
                 global_config: dict,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: dict[str, multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            name="linear" + "_" + str(task_id),
            worker_type="linear",
            worker_idx=task_id,
            global_config=global_config,
            config=config,
            queues=queues,
            run_name=run_name,
        )

        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )

    async def solve(self, game: LeanGame) -> None:
        agent: LazyLeanAgent = LazyLeanAgent(
            game=game,
            worker=self,
            max_num_completions=1  # For linear inference, we only need one completion
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

    def loop(self):
        pass
