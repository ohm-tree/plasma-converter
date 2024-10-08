"""
game.py

This module contains the Game class, the abstract base class for any one-player,
perfect information, abstract strategy game.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Iterable, List, TypeVar

import numpy as np

from src.games.concurrent import ConcurrentClass
from workers.worker import WorkerResponse, WorkerTask

GameMove = TypeVar('GameMove')


class ConcurrentGameState(ConcurrentClass, ABC, Generic[GameMove]):
    """
    The ConcurrentGameState class is an abstract base class for representing the state of a game.
    Derived classes need to implement the specific state representation and methods.
    """

    @classmethod
    @abstractmethod
    def saves(cls, states: List['ConcurrentGameState'], filename: str):
        """
        Save a collection of states to a file.

        This is super important. During MCTS,
        this is what gets saved to the replay buffer.

        Parameters
        ----------
        states : List[ConcurrentGameState]
            A list of game states to save.
        filename : str
            The name of the file to save the states to.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def loads(cls, filename: str) -> List['ConcurrentGameState']:
        """
        Load a collection of states from a file.

        Parameters
        ----------
        filename : str
            The name of the file to load the states from.
        """
        raise NotImplementedError

    @abstractmethod
    def ready(self) -> bool:
        """
        Returns True if the game state is ready (terminal, reward, and nextstate can be called).
        """
        raise NotImplementedError

    @abstractmethod
    def next_state(self, action: GameMove) -> 'ConcurrentGameState':
        """
        Returns the next state of the game given a current state and action.
        Requires that the state is non-terminal and action is legal.

        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Returns True if the game is over, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def reward(self) -> float:
        """
        Returns a float consisting of the reward for the player.
        """
        raise NotImplementedError

    @abstractmethod
    def display_state(self) -> str:
        """
        Returns a string representation of the game state.
        """
        raise NotImplementedError

    @abstractmethod
    def __hash__(self) -> int:
        """
        Returns a hash of the game state.
        """
        raise NotImplementedError


ConcurrentGameStateType = TypeVar(
    'ConcurrentGameStateType', bound=ConcurrentGameState)


class ConcurrentMetaGameState(ConcurrentGameState[GameMove], ABC):
    @abstractmethod
    def rollout_ready(self) -> bool:
        """
        Returns True if the game state is ready for a policy and value query.
        """
        raise NotImplementedError

    @abstractmethod
    def policy(self) -> np.ndarray:
        """
        Returns the policy for the game state.
        """
        raise NotImplementedError

    @abstractmethod
    def value(self) -> float:
        """
        Returns the value for the game state.
        """
        raise NotImplementedError

    @abstractmethod
    def active_moves(self) -> Iterable[GameMove]:
        """
        Returns the active moves for the game state.
        """
        raise NotImplementedError

    @abstractmethod
    def index_active_move(self, move: GameMove) -> int:
        """
        Returns the index of the active move in the game state.
        """
        raise NotImplementedError

    @abstractmethod
    def len_active_moves(self) -> int:
        """
        Returns the number of active moves in the game state.
        """
        raise NotImplementedError

    @abstractmethod
    def require_new_move(self):
        """
        Demand to see a new move.
        """
        raise NotImplementedError
