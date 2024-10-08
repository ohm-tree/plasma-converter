"""
uct_tree.py

This module contains the UCTNode class, which represents a node in the UCT search tree.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

from typing import Dict, Generic, Optional

import numpy as np

from games.lean_game import LeanGame, LeanGameState
from src.uct.uct_node import UCTNode


class UCTTree:

    def __init__(self, game: LeanGame, game_state: LeanGameState, action: int, parent: 'UCTNode' = None, init_type: str = "zero"):
        """
        Initialize a new Lazily-Processed UCTTree.
        """
        self.root = UCTNode(game, game_state, action, parent, init_type)
        self.game = game

        self.init_type = init_type
