"""
uct_node.py

This module contains the UCTNode class, which represents a node in the UCT search tree.
The code is adapted from https://www.moderndescartes.com/essays/deep_dive_mcts/.
"""

from typing import Dict, Generic, Optional

import numpy as np

from src.games.game import Game, GameState


class UCTNode:

    def __init__(self, game: Game, game_state: GameState, action: int, parent: 'UCTNode' = None, init_type: str = "zero"):
        """
        Initialize a new UCTNode.
        """
        self.game: Game = game
        self.game_state: GameState = game_state

        # Action to enter the node, -1 if root
        self.action: int = action

        # Parent of the node, None if root
        self.parent: Optional[UCTNode] = parent

        # This is a dictionary of action -> UCTNode. Only legal actions are keys
        self.children: Dict[int, UCTNode] = {}

        # Whether the node has been expanded to its children
        self.is_expanded: bool = False

        # Whether the node has been processed.
        self.is_processed: bool = False

        # Cached values so that we don't need to recompute them every time
        self._action_mask = None
        self._is_terminal = None

        # The priors and values are obtained from a neural network every time you expand a node
        # The priors, total values, and number visits will be 0 on all illegal actions
        num_actions = len(self.action_mask)

        # These values are initialized in the expand() function.
        self.child_priors: np.ndarray = np.zeros(num_actions)
        self.child_total_value: np.ndarray = np.zeros(num_actions)
        self.child_number_visits: np.ndarray = np.zeros(num_actions)

        # Used iff you are the root.
        if self.parent is None:
            self.root_total_value: float = 0
            self.root_number_visits: int = 0

        self.init_type = init_type

    def __hash__(self):
        """
        Hash the node based on the game state.
        """
        return hash(self.game_state)

    def root(self):
        """
        Set the parent of the node to None.
        """
        if self.parent is not None:
            # I was not originally the root, so actually I have some data to pass down.
            self.root_total_value = self.total_value
            self.root_number_visits = self.number_visits
        else:
            # I am the root, and I never had a parent, so I should initialize these values.
            self.root_total_value = 0
            self.root_number_visits = 0
        self.parent = None
        self.action = -1

    @property
    def action_mask(self):
        if self._action_mask is None:
            self._action_mask = self.game.action_mask(self.game_state)
        return self._action_mask
    
    @property
    def is_terminal(self):
        if self._is_terminal is None:
            self._is_terminal = self.game.is_terminal(self.game_state)
        return self._is_terminal

    @property
    def number_visits(self):
        if self.parent is None:
            return self.root_number_visits
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        if self.parent is None:
            self.root_number_visits = value
        else:
            self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        if self.parent is None:
            return self.root_total_value
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        if self.parent is None:
            self.root_total_value = value
        else:
            self.parent.child_total_value[self.action] = value

    def child_Q(self):
        """
        The value estimate for each child, based on the average value of all visits.
        """
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        """
        The uncertainty for each child, based on the UCT formula (think UCB).
        """
        return np.sqrt(1 + np.sum(self.child_number_visits)) * (self.child_priors / (1 + self.child_number_visits))

    def best_child(self, c: float):
        """
        Compute the best legal child, with the action mask.
        """
        scores = self.child_Q() + c * self.child_U()

        # mask out illegal actions
        scores[~self.action_mask] = -np.inf

        # print("scores", scores)
        # print("action mask", self.action_mask)

        return self.children[np.argmax(scores)]

    def select_leaf_no_virtual_loss(self, c: float = 1.0) -> 'UCTNode':
        """
        Deterministically select the next leaf to expand based on the best path.
        """
        current = self

        # iterate until either you reach an un-expanded node or a terminal state
        while (
            current.is_expanded and
            current.is_processed and
            (not current.is_terminal)
        ):
            current = current.best_child(c)

        return current

    def select_leaf(self, c: float = 1.0) -> 'UCTNode':
        """
        Deterministically select the next leaf to expand based on the best path.
        """
        current = self

        assert self.is_expanded, "Cannot select a leaf from an un-expanded node."
        assert self.is_processed, "Cannot select a leaf from an un-processed node."
        assert not self.is_terminal, "Cannot select a leaf from a terminal node."

        # iterate until either you reach an un-expanded node or a terminal state
        while (
            current.is_expanded and
            current.is_processed and
            (not current.is_terminal)
        ):

            # Add a virtual loss.
            current.number_visits += 1
            current.total_value -= 1
            current = current.best_child(c)

        # Add a virtual loss.
        current.number_visits += 1
        current.total_value -= 1

        return current

    def expand(self, child_priors, value_estimate, train=True):
        """
        Expand a non-terminal, un-expanded node using the child_priors from the neural network.
        """
        assert not self.game.is_terminal(
            self.game_state), "Cannot expand a terminal node."
        assert not self.is_expanded, "Cannot expand an already expanded node."

        # if train and self.action == -1:
        #     # if you are the root, mix dirichlet noise into the prior
        #     child_priors = 0.75 * child_priors + 0.25 * \
        #         dirichlet_noise(self.action_mask, 0.3)

        # Recommended by minigo which cites Leela.
        # See https://github.com/tensorflow/minigo/blob/master/cc/mcts_tree.cc line 448.
        # See https://www.reddit.com/r/cbaduk/comments/8j5x3w/first_play_urgency_fpu_parameter_in_alpha_zero/

        if self.init_type == "offset":
            self.child_total_value += (value_estimate - 0.1) * self.action_mask
        if self.init_type == "equal":
            self.child_total_value += value_estimate * self.action_mask
        if self.init_type == "invert":
            self.child_total_value += (- value_estimate) * self.action_mask
        if self.init_type == "zero":
            pass

        for action, prior in enumerate(child_priors):
            if self.action_mask[action]:
                self.add_child(action, prior)
                # assert action in self.children

        self.is_expanded = True

    def add_child(self, action, prior):
        """
        Add a child with a given action and prior probability.

        The value_estimates are updated together in expand().
        """
        assert not action in self.children, f"Child with action {action} already exists."

        self.child_priors[action] = prior

        self.children[action] = UCTNode(
            self.game,
            self.game.next_state(self.game_state, action),
            action,
            parent=self,
            init_type=self.init_type
        )

    def backup(self, estimate):
        """
        Propagate the estimate of the current node,
        back up along the path to the root.
        """
        current = self
        while current.parent is not None:
            # Do not increment the number of visits here, because it is already done in select_leaf.
            # Extra +1 to the estimate to offset the virtual loss.
            current.total_value += estimate + 1
            current = current.parent


def dirichlet_noise(action_mask, alpha):
    """
    Returns a dirichlet noise distribution on the support of an action mask.
    """

    places = action_mask > 0
    noise = np.random.dirichlet(alpha * np.ones(np.sum(places)))

    noise_distribution = np.zeros_like(action_mask, dtype=np.float64)
    noise_distribution[places] = noise

    return noise_distribution
