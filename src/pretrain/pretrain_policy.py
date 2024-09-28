import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple
import argparse
import os
import wandb
import sys
import yaml
from pathlib import Path

from src.networks.prover_llm import ProverLLM


def save_checkpoint(state, checkpoint_dir, epoch):
    """
    Saves the training checkpoint.

    Parameters:
    ----------
    state: dict
        A dictionary containing model state, optimizer state, epoch, etc.
    checkpoint_dir: str
        Directory where checkpoints will be saved.
    epoch: int
        Current epoch number.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Loads the training checkpoint.

    Parameters:
    ----------
    model: YourModelClass
        The neural network model.
    optimizer: torch.optim.Optimizer
        The optimizer.
    checkpoint_path: str
        Path to the checkpoint file.
    device: torch.device
        Device to map the checkpoint tensors.
    
    Returns:
    -------
    epoch: int
        The epoch to resume from.
    """
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.policy_head.load_state_dict(checkpoint['policy_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {epoch + 1}")
        return epoch + 1
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0

def train_policy(model: ProverLLM,
                 data_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module = nn.CrossEntropyLoss(),
                 device: str = 'cuda',
                 num_epochs: int = 10,
                 checkpoint_dir: str = "",
                 start_epoch=0):
    model.to(device)
    model.train()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            # Assume that each batch is a tuple of (intermediate_output, target_policy)
            intermediate_output, target_policy = batch
            intermediate_output = intermediate_output.to(device)
            target_policy = target_policy.to(device)

            optimizer.zero_grad()

            # Get policy output
            policy_output, _ = model.policy_and_value(intermediate_output)

            # Compute loss
            loss = criterion(policy_output, target_policy)

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
                wandb.log({"batch_loss": loss.item()})

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "average_loss": avg_loss})

        # Save checkpoint at the end of each epoch
        checkpoint_state = {
            'epoch': epoch,
            'policy_head_state_dict': model.policy_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Add other necessary components if needed
        }
        save_checkpoint(checkpoint_state, checkpoint_dir, epoch + 1)

    print("Training completed.")

def main():
    parser = argparse.ArgumentParser(description="Train Policy Head of Neural Network with YAML Config, Checkpointing, and wandb")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Load configuration from YAML
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize wandb
    wandb.init(project=config['wandb']['project'], 
               entity=config['wandb'].get('entity', None), 
               config=config,
               name=config['wandb'].get('run_name', None))
    wandb_config = wandb.config

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    wandb.config.device = str(device)

    # Initialize model
    if config['model']['pretrained_model_path'] is None:
        model = ProverLLM(random_flag = True)
    else:
        model = ProverLLM()
        model.load_state_dict(torch.load(config['model']['pretrained_model_path'], map_location=device))

    model.to(device)

    # Freeze value_head parameters
    if config['model'].get('freeze_value_head', True):
        for param in model.value_head.parameters():
            param.requires_grad = False

    # Prepare data loader
    # Replace this with your actual data loading logic
    # For example, if you have a custom Dataset:
    # from your_dataset_module import YourDataset
    # dataset = YourDataset(config['data']['data_path'])
    # data_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)

    try:
        from src.pretrain.utils import load_workbook_problems  # Replace with your actual dataset module
        dataset = load_workbook_problems(config['data']['data_path'])
        data_loader = DataLoader(dataset, 
                                 batch_size=config['training']['batch_size'], 
                                 shuffle=True, 
                                 num_workers=config['data'].get('num_workers', 4))
    except ImportError:
        print("Please implement the data loading logic by defining YourDataset in your_dataset_module.")
        sys.exit(1)

    # Define optimizer (only for policy_head parameters)
    optimizer = optim.Adam(model.policy_head.parameters(), lr=config['training']['learning_rate'])

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Checkpointing
    checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')
    resume_checkpoint = config['training'].get('resume_checkpoint', None)
    start_epoch = 0
    if resume_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, resume_checkpoint, device)

    # Train the policy head
    train_policy(model, data_loader, optimizer, criterion, device, 
                config['training']['num_epochs'], checkpoint_dir, start_epoch)

    # Save the trained policy_head
    save_path = config['training'].get('save_path', 'policy_head.pth')
    torch.save(model.policy_head.state_dict(), save_path)
    print(f"Trained policy head saved to {save_path}")
    wandb.save(save_path)

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()