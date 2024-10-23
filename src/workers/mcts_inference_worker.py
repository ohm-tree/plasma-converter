"""
This file contains the main entrypoint for the MCTS algorithm.
It is responsible for running the MCTS algorithm on a given problem,
saving the results to disk, and logging the results.
"""

import multiprocessing

from wayfinder.games.agent import Agent
from wayfinder.uct.self_play import async_self_play

from src.agents.lazy_agent import LazyLeanAgent
from src.lean.lean_game import LeanGame, LeanMove, LeanState
from src.workers.inference_worker import InferenceWorker


class MCTSWorker(InferenceWorker):
    def __init__(self,
                 global_config: dict,
                 config: dict,
                 run_name: str,
                 task_id: int,
                 queues: dict[str, multiprocessing.Queue],
                 **kwargs  # Unused
                 ):
        super().__init__(
            name="mcts" + "_" + str(task_id),
            worker_type="mcts",
            worker_idx=task_id,
            global_config=global_config,
            config=config,
            queues=queues,
            run_name=run_name,
        )

        if "tree_kwargs" not in self.config:
            self.config["tree_kwargs"] = {}

        if "search_kwargs" not in self.config:
            self.config["search_kwargs"] = {}

        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )

    def create_agent(self, game: LeanGame) -> Agent[LeanGame, LeanState, LeanMove]:
        if self.config['agent_class'] == 'LazyLeanAgent':
            return LazyLeanAgent(
                game=game,
                worker=self,
                **self.config['agent_kwargs']
            )

        raise ValueError(f"Unknown agent class: {self.config['agent_class']}")

    async def solve(self, game: LeanGame) -> dict:
        state: LeanState = await game.starting_state()

        agent = self.create_agent(game)

        return await async_self_play(
            self.logger,
            state=state,
            game=game,
            agent=agent,
            tree_kwargs=self.config['tree_kwargs'],
            search_kwargs=self.config['search_kwargs']
        )

    def loop(self):
        pass
