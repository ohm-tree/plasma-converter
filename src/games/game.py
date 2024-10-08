"""
game.py

This module contains the Game class, the abstract base class for any one-player,
perfect information, abstract strategy game.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Hashable, Iterable, List, TypeVar

import numpy as np

from src.games.concurrent import ConcurrentClass
from workers.worker import WorkerResponse, WorkerTask

GameMove = TypeVar('GameMove', bound=Hashable)


class ConcurrentGameState(ConcurrentClass, ABC, Generic[GameMove], Hashable):
    """
    The ConcurrentGameState class is an abstract base class
    for representing the state of a game.
     - These games are **difficult to compute**; have long non-cpu-bound
    computation steps which can be run concurrently.
     - These games have **large branching factor**; the number of possible
    moves is large and represented by a generate GameMove instead of an index (say).

    To this end, a ConcurrentGameState is not ready() upon initialization.
    When a ConcurrentGameState startup is called (either in __init__ with
    startup = True or with startup()), it sets off a long concurrent computation
    which ends with ready() returning True.

    Derived classes need to implement the specific state representation and methods.

    self.ready() returns True if the game state is ready
    (terminal, reward, and nextstate can be called).
    """

    def __init__(self, worker_id: Any):
        super().__init__(worker_id)

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

    @classmethod
    @abstractmethod
    def starting_state(cls, *args, **kwargs) -> 'ConcurrentGameState':
        """
        Returns the starting state of the game.
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
    def terminal(self) -> bool:
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
    def make_root(self) -> 'ConcurrentGameState':
        """
        Severs any references to the parent of
        this state, making it the root of the tree.
        """
        raise NotImplementedError


ConcurrentGameStateType = TypeVar(
    'ConcurrentGameStateType', bound=ConcurrentGameState)

MetaGameMove = TypeVar('MetaGameMove', bound=Hashable)


class ConcurrentMetaGameState(ConcurrentGameState, ABC, Generic[ConcurrentGameStateType, MetaGameMove], Hashable):
    """
    The ConcurrentMetaGameState class is an abstract base class
    for representing the state of a game plus the internal state of an agent
    which is playing the game.

    A ConcurrentMetaGameState contains a ConcurrentGameState. Concurrent operations
    will occur while a ConcurrentMetaGameState is not ready(). Once
    ready() is true, the ConcurrentMetaGameState will be ready for:
     - a policy and value query for all active_moves().
     - next_state calls for all active_moves().

    If at some point this subset active_moves() is deemed insufficient for
    representing the full universe of legal moves,
    one can call require_new_move() to demand a new move; this resets ready()
    to false.

    A ConcurrentMetaState is also a ConcurrentGameState itself, which exposes the terminal()
    and reward() functions of its internal game state. Note that the ready() function of
    a ConcurrentMetaGameState is not the same as the ready() function of its internal game state;
    in addition to the internal state being ready, the agent must also be ready to transition
    to a new game state.
    """

    def __init__(self, worker_id: Any, internal_state: ConcurrentGameStateType = None):
        super().__init__(worker_id=worker_id)
        self.state = internal_state

    def terminal(self) -> bool:
        return self.state.terminal()

    def reward(self) -> float:
        return self.state.reward()

    def internal_ready(self) -> bool:
        return self.state.ready()

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
