import torch
import torch.nn as nn
import torch.nn.functional as F

from src.games.sudoku import SudokuGame
from src.networks.network import Network


class SudokuCNN(Network):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        # Output heads
        # Action head: Outputs probabilities for each position and number
        self.action_head = nn.Conv2d(128, 9, kernel_size=1)
        # Value head: Outputs a single scalar value
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(81, 1)
        )

    def forward(self, x):
        # Input x should be of shape
        # (batch_size, number, row, column) = (b, 9, 9, 9).

        # Convolutional layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Two heads output
        actions = self.action_head(x)
        # reshape actions to (729,)
        # (batch_size, number, row, column)
        actions = actions.reshape(-1, 729)  # (batch_size, 729)
        value = self.value_head(x)

        print(actions.shape, value.shape)

        return actions, value


# Example usage:
# Assuming a batch size of 1 for simplicity
model = SudokuCNN()
example_input = torch.rand(3, 9, 9, 9)  # Random input tensor
actions, value = model(example_input)

print("Actions shape:", actions.shape)  # Expected shape: (1, 9, 9, 9)
print("Value shape:", value.shape)  # Expected shape: (1, 1)
