"""
This file contains the main entrypoint for the MCTS algorithm.
It is responsible for running the MCTS algorithm on a given problem,
saving the results to disk, and logging the results.
"""

import multiprocessing

from wayfinder.uct.self_play import async_self_play

from src.games.lazy_agent import LazyLeanAgent
from src.games.lean_game import LeanGame, LeanState
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

        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )

    async def solve(self, game: LeanGame) -> None:
        state: LeanState = await game.starting_state()

        states: list[LeanState]

        agent = LazyLeanAgent(
            game=game,
            worker=self,
            max_num_completions=self.config['max_num_completions'],
        )

        states, distributions, rewards = await async_self_play(
            self,
            state=state,
            game=game,
            agent=agent,
            tree_kwargs=self.config['tree_kwargs'],
            search_kwargs=self.config['search_kwargs']
        )

        return {
            'states': states,
            'distributions': distributions,
            'rewards': rewards,
            "result": rewards[-1]
        }

    def loop(self):
        pass
