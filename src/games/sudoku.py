from typing import List, Optional

import numpy as np

from src.games.game import LeanGame


class SudokuGameState:
    def __init__(self, board: np.ndarray, is_dead: bool = False):
        self.board = board
        self.is_dead = is_dead

    def from_string(s: str) -> 'SudokuGameState':
        """
        Create a SudokuGameState from a string representation of the board,
        which is a 81-length string of '0' through '9,' where '0' represents an empty cell.
        """
        board = np.array([int(c) for c in s]).reshape((9, 9))
        return SudokuGameState(board)

    def to_numpy(self) -> np.ndarray:
        """
        Convert the game state to a tensor (n, r, c).
        game.state.board is currently a numpy array of size (r, c) = (9, 9) with values from 0 to 9.
        This should be one-hot encoded to shape (9, 9, 9).
        The one hot takes in a long tensor, so we need to convert it to a tensor with dtype long.
        Then, one_hot will create a tensor of shape (r, c, n) = (9, 9, 10), we permute it to (10, 9, 9),
        and slice it to (9, 9, 9).
        Finally, we convert it to float because one_hot returns a tensor of dtype long.
        """
        one_hot = np.zeros((9, 9, 10), dtype=int)  # (r, c, n + 1)
        one_hot[np.arange(9)[:, None], np.arange(9), self.board] = 1
        one_hot = one_hot[:, :, 1:].astype(float)  # (r, c, n)
        one_hot = np.transpose(one_hot, (2, 0, 1))  # (n, r, c)
        return one_hot

    def __str__(self):
        return '\n'.join(' '.join(str(cell) if cell != 0 else '.' for cell in row) for row in self.board)


class SudokuGame(LeanGame[SudokuGameState]):
    """
    A Sudoku game implementation of the Game class.
    """

    def action_to_tuple(self, action: int) -> tuple:
        """
        Converts an action (0-80) to a tuple (number, row, col).
        """
        number = action // 81
        row = (action % 81) // 9
        col = action % 9
        return row, col, number + 1

    def tuple_to_action(self, row: int, col: int, number: int) -> int:
        """
        Converts a tuple (row, col, number) to an action (0-728).
        """
        return (number - 1) * 81 + row * 9 + col

    def next_state(self, state: SudokuGameState, action: int) -> SudokuGameState:
        """
        Places a number on the Sudoku board.
        Action is a tuple: (row, col, number)
        Checks if the move made is valid, if not, marks the state as dead.
        """
        row, col, number = self.action_to_tuple(action)
        new_board = np.copy(state.board)
        new_board[row, col] = number

        # Check if the move is legal only for the new move
        if not self.is_action_legal(state, row, col, number):
            return SudokuGameState(board=new_board, is_dead=True)

        # if there are no legal actions in the action_mask, then set is_dead to True
        if np.sum(self.action_mask(SudokuGameState(board=new_board))) == 0:
            return SudokuGameState(board=new_board, is_dead=True)

        return SudokuGameState(board=new_board)

    def action_mask(self, state: SudokuGameState) -> np.ndarray:
        # # return all 1s.
        # return np.ones((729,), dtype=int)

        result = np.zeros((729,), dtype=np.bool)
        for row in range(9):
            for col in range(9):
                for number in range(1, 10):
                    if self.is_action_legal(state, row, col, number):
                        result[self.tuple_to_action(row, col, number)] = 1

        return result

    def is_terminal(self, state: SudokuGameState) -> bool:
        """
        The game is considered over if the board is fully filled or if the state is dead.
        """
        return state.is_dead or np.all(state.board != 0)

    def reward(self, state: SudokuGameState) -> float:
        """
        Rewards the player if the board is correctly completed, otherwise 0.
        """
        assert self.is_terminal(
            state), "Reward can only be calculated for terminal states."
        if np.all(state.board != 0):
            return 1.0
        if state.is_dead:
            return -1.0
        return 1.0

    def display_state(self, state: SudokuGameState) -> str:
        """
        Returns a string representation of the game state, marking dead states.
        """
        board_string = '\n'.join(' '.join(
            str(cell) if cell != 0 else '.' for cell in row) for row in state.board)
        if state.is_dead:
            board_string += "\nState is dead due to an invalid move."
        return board_string

    def hash_state(self, state: SudokuGameState) -> int:
        """
        Returns a hash of the game state.
        """
        return hash(state.board.tobytes())

    def is_action_legal(self, state: SudokuGameState, row: int, col: int, number: int) -> bool:
        """
        Determines if placing a number in the given row and column is legal, assuming the move just made.
        """
        # Check if the position was already filled
        if state.board[row, col] != 0:
            return False

        # Check row, column, and 3x3 grid for the number
        if number in state.board[row] or number in state.board[:, col]:
            return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if number in state.board[start_row:start_row+3, start_col:start_col+3]:
            return False
        return True

    def is_board_valid(self, board: np.ndarray) -> bool:
        """
        Check if the entire board is valid. This never needs to be called,
        because we check legality on every move, so at all times the board
        is valid.
        """
        for i in range(9):
            if not self.is_group_valid(board[i]) or not self.is_group_valid(board[:, i]):
                return False
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                if not self.is_group_valid(board[row:row+3, col:col+3].flatten()):
                    return False
        return True

    def is_group_valid(self, group: np.ndarray) -> bool:
        """
        Checks if a group (row, column, or block) contains no duplicates of numbers 1-9.
        """
        return np.all(np.bincount(group, minlength=10)[1:] <= 1)
