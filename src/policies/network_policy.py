"""
network_policy.py

This module contains the NetworkPolicy class, a basic implementation
of a Policy class that uses a neural network.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from src.games.game import Game, GameState
from src.networks.network import Network
from src.policies.policy import Policy


class NetworkPolicy(Policy):
    """
    A Policy that uses a neural network to output a probability distribution over actions.
    """

    def __init__(self, network: Network, device: Optional[torch.device] = None):
        self.network = network
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.network.to(self.device)

    def action(self, game: Game, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Outputs a probability distribution over valid actions and a value estimate given a game and state.

        Parameters:
        ----------
        game: Game
            The game instance.
        state: GameState
            The current state of the game.

        Returns:
        -------
        Tuple[np.ndarray, float]
            A tuple containing the probability distribution over actions and the value estimate.
        """

        policy, value = self.network.forward(state)

        policy = policy.squeeze(0)  # Remove batch dimension
        value = value.squeeze(0)

        # the network outputs logits, so we need to apply softmax
        policy = torch.nn.functional.softmax(policy, dim=0)
        policy = policy.detach().numpy()
        value = value.detach().numpy()

        policy *= game.action_mask(state)  # mask out illegal actions
        if np.sum(policy) <= 0:
            # if policy is too small, output uniform distribution
            policy = game.action_mask(state)

        policy /= np.sum(policy)

        return policy, value
