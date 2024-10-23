"""
This file contains the main entrypoint for the MCTS algorithm.
It is responsible for running the MCTS algorithm on a given problem,
saving the results to disk, and logging the results.
"""

import multiprocessing

from wayfinder.uct.self_play import async_self_play

from src.agents.lazy_agent import LazyLeanAgent
from src.agents.lazy_valueless_agent import LazyValuelessLeanAgent
from src.lean.lean_game import LeanGame, LeanState
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

    async def solve(self, game: LeanGame) -> dict:
        state: LeanState = await game.starting_state()

        states: list[LeanState]

        # TODO: add config flag for switching between valueless and regular agents
        agent = LazyValuelessLeanAgent(
            game=game,
            worker=self,
            max_num_completions=self.config['max_num_completions'],
        )

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
