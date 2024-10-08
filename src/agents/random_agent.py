"""
random_agent.py

An agent that samples actions randomly.
"""

import numpy as np

from src.agents.agent import Agent
from src.games.game import GameState, LeanGame


class RandomAgent(Agent):
    """
    An agent that samples actions uniformly at random.
    """

    def action(self, game: LeanGame, state: GameState) -> int:
        action_mask = game.action_mask(state)
        return np.random.choice(np.arange(len(action_mask)), p=action_mask / np.sum(action_mask))
