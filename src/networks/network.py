from abc import ABC, abstractmethod
from typing import Tuple

import torch

from src.games.game import GameState


class Network(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x: GameState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the forward pass of the network. Returns a policy and a value.
        """
