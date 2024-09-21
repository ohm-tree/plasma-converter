"""
game.py

This module contains the Game class, the abstract base class for any one-player,
perfect information, abstract strategy game.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, TypeVar

import numpy as np


class GameState(ABC):
    """
    The GameState class is an abstract base class for representing the state of a game.
    Derived classes need to implement the specific state representation and methods.
    """

    @classmethod
    @abstractmethod
    def saves(cls, states: List['GameState'], filename: str):
        """
        Save a collection of states to a file.

        This is super important. During MCTS,
        this is what gets saved to the replay buffer.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def loads(cls, filename: str) -> List['GameState']:
        """
        Load a collection of states from a file.
        """
        raise NotImplementedError


GameStateType = TypeVar('GameStateType', bound=GameState)

class Game(ABC, Generic[GameStateType]):
    """
    The Game class is the abstract base class for any one-player,
    perfect information, abstract strategy game.

    Derived classes need to implement game logic, symmetries, and action masks.

    Methods:
        next_state: returns next state of the game given current state and action
        is_terminal: returns True if the game is over, False otherwise
        action_mask: returns a mask of legal actions for the player
        reward: returns the rewards for the player
        display_state: returns a string representation of the game state
        hash_state: returns a hash of the game state
    """

    @abstractmethod
    def next_state(self, state: GameStateType, action: int) -> GameStateType:
        """
        Returns the next state of the game given a current state and action.
        Requires that the state is non-terminal and action is legal.

        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: GameStateType) -> bool:
        """
        Returns True if the game is over, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def action_mask(self, state: GameStateType) -> np.ndarray:
        """
        Returns a mask of legal actions for the player.

        Requires that the state is non-terminal.

        Make sure that the np.ndarray dtype is np.bool!!!! Otherwise negation will produce unexpected results.
        """
        raise NotImplementedError

    @abstractmethod
    def reward(self, state: GameStateType) -> float:
        """
        Returns a float consisting of the reward for the player.
        """
        raise NotImplementedError

    @abstractmethod
    def display_state(self, state: GameStateType) -> str:
        """
        Returns a string representation of the game state.
        """
        raise NotImplementedError

    @abstractmethod
    def hash_state(self, state: GameStateType) -> int:
        """
        Returns a hash of the game state.
        """
        raise NotImplementedError
