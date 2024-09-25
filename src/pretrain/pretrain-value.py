'''
The value network is a simple feedforward neural network that takes in the state
(i.e. the tactic state of the lean proof)
and outputs a scalar value that represents the value of the state.

First, we will pretrain the value network on a dataset of lean proofs, so that
it learns to predict *the number of steps required to complete the proof after this state*.
'''


from utils import load_data
from src.networks.prover_llm import ProverLLM
import torch
import torch.nn as nn

num_samples = 1
data = load_data(num_samples = num_samples)

print(len(data))

# use tokenize, get_intermediate_state, and value_head t