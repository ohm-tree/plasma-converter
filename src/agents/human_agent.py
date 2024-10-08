"""
human_agent.py

An agent that allows a human to play a game interactively.
"""

import numpy as np

from src.agents.agent import Agent
from src.games.game import GameState, LeanGame


def get_human_action(action_mask: np.ndarray) -> int:
    """
    Get a human action from the command line.
    """
    while True:
        try:
            action = int(input("Enter an action: "))
            if 0 <= action < len(action_mask) and action_mask[action]:
                return action
            else:
                print("Invalid action. Please enter a legal action.")

        except ValueError:
            print("Invalid input. Please enter an integer.")


class HumanAgent(Agent):
    """
    An agent that allows a human to play a game interactively.
    """

    def action(self, game: LeanGame, state: GameState) -> int:
        """
        Outputs an action given a game and state, by asking the human to input an action.
        """
        action_mask = game.action_mask(state)
        print(f"Legal action mask: {action_mask}")

        return get_human_action(action_mask)
