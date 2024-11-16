"""
This file contains the main entrypoint for the MCTS algorithm.
It is responsible for running the MCTS algorithm on a given problem,
saving the results to disk, and logging the results.
"""

import multiprocessing
from multiprocessing import sharedctypes
from typing import Union

from wayfinder.games.agent import Agent
from wayfinder.uct.self_play import async_self_play

from src.agents.lazy_agent import LazyLeanAgent
from src.agents.scratchpad_lazy_agent import ScratchpadLazyAgent
from src.lean.lean_game import LeanGame, LeanMove, LeanState
from src.lean.scratchpad_lean_game import (
    ScratchpadGame,
    ScratchpadMove,
    ScratchpadState,
)
from src.workers.inference_worker import InferenceWorker


class MCTSWorker(InferenceWorker):
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
            name="mcts" + "_" + str(task_id),
            worker_type="mcts",
            worker_idx=task_id,
            global_config=global_config,
            config=config,
            queues=queues,
            values=values,
            run_name=run_name,
        )

        if "tree_kwargs" not in self.config:
            self.config["tree_kwargs"] = {}

        if "search_kwargs" not in self.config:
            self.config["search_kwargs"] = {}

        self.global_config = global_config
        self.game_class = self.config['game_class']

        self.logger.info(
            f"Global Variables I can see: {globals().keys()}"
        )

    def create_lean_agent(self, game: LeanGame) -> Agent[LeanGame, LeanState, LeanMove]:
        if self.config['agent_class'] == 'LazyLeanAgent':
            return LazyLeanAgent(
                game=game,
                worker=self,
                **self.config['agent_kwargs']
            )
        raise ValueError(f"Unknown agent class: {self.config['agent_class']}")

    def create_scratchpad_agent(self, game: ScratchpadGame) -> Agent[ScratchpadGame, ScratchpadState, ScratchpadMove]:
        return ScratchpadLazyAgent(
            game=game,
            worker=self,
            **self.config['agent_kwargs']
        )

    async def solve(self, game: Union[LeanGame, ScratchpadGame]) -> dict:
        if isinstance(game, LeanGame):
            assert self.game_class == 'lean'
            return await self.solve_lean(game)
        elif isinstance(game, ScratchpadGame):
            assert self.game_class == 'scratchpad'
            return await self.solve_scratchpad(game)
        raise ValueError(f"Unknown game class: {type(game)}")

    async def solve_lean(self, game: LeanGame) -> dict:
        state: LeanState = await game.starting_state()

        agent = self.create_lean_agent(game)

        return await async_self_play(
            self.logger,
            state=state,
            game=game,
            agent=agent,
            tree_kwargs=self.config['tree_kwargs'],
            search_kwargs=self.config['search_kwargs']
        )

    async def solve_scratchpad(self, game: ScratchpadGame) -> dict:
        state: ScratchpadState = await game.starting_state()
        agent = self.create_scratchpad_agent(game)
        return await async_self_play(
            self.logger,
            state=state,
            game=game,
            agent=agent,
        )

    def loop(self):
        pass
