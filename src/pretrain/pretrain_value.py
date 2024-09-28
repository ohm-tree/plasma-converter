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

samples_idx = [0]
data = load_data(samples_idx = samples_idx)

print("bonk:")
print(data[0].segmentation())

# use tokenize, get_intermediate_state, and value_head to turn the lean text into numbers
# then feed it through the value network and train

