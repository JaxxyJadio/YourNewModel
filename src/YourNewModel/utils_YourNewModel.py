"""
Utility functions for machine learning model training.

This module provides a collection of helper functions for common tasks
in deep learning workflows, including saving and loading model checkpoints,
clipping gradients to prevent exploding gradients, and logging training metrics.
"""

import os
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
from typing import Any, Dict, Optional

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str) -> None:
    """
    Saves the current state of the model, optimizer, and training epoch to a file.

    This function is crucial for resuming training from a specific point,
    preventing loss of progress due to interruptions, or for storing
    trained models for later inference.

    Args:
        model (torch.nn.Module): The PyTorch model whose state_dict is to be saved.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer whose state_dict
                                           is to be saved. This allows resuming
                                           optimizer state (e.g., learning rates).
        epoch (int): The current training epoch number.
        path (str): The file path where the checkpoint will be saved.
                    Example: 'checkpoints/model_epoch_10.pt'
    """
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    # Ensure the directory exists before saving
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path_obj))
    print(f"Checkpoint saved to {path_obj} at epoch {epoch}")

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    """
    Loads the model and optimizer states from a checkpoint file.

    This function is used to restore a model and optimizer to a previous state,
    typically for resuming interrupted training or for fine-tuning.

    Args:
        model (torch.nn.Module): The PyTorch model instance to which the loaded
                                 state_dict will be applied.
        optimizer (torch.optim.Optimizer): The PyTorch optimizer instance to which
                                           the loaded state_dict will be applied.
        path (str): The file path from which the checkpoint will be loaded.

    Returns:
        int: The epoch number recorded in the loaded checkpoint, indicating
             the point at which training was last saved.

    Raises:
        FileNotFoundError: If the specified checkpoint `path` does not exist.
        RuntimeError: If there's an issue loading the state_dict (e.g., mismatch
                      between model architecture and saved state).
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path_obj}")
    checkpoint = torch.load(str(path_obj))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path_obj}. Resuming from epoch {checkpoint['epoch']}.")
    return checkpoint['epoch']

def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """
    Clips the gradients of the model's parameters to a maximum norm.

    This technique is commonly used to prevent the "exploding gradient" problem,
    where gradients become excessively large during training, leading to unstable
    updates and divergence.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters' gradients
                                 will be clipped.
        max_norm (float): The maximum norm for the gradients. If the total norm
                          of gradients exceeds this value, they will be scaled down.

    Returns:
        float: The total norm of the gradients before clipping. This can be useful
               for monitoring gradient magnitudes during training.
    """
    # clip_grad_norm_ modifies gradients in-place and returns the total norm
    total_norm = clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)

def log_metrics(metrics: Dict[str, Any], step: int) -> None:
    """
    Logs a dictionary of metrics to standard output.

    This function provides a basic logging mechanism for tracking training
    and evaluation metrics at specific steps. It can be easily extended
    to integrate with more advanced logging tools like TensorBoard, Weights & Biases,
    or MLflow for richer visualization and experiment tracking.

    Args:
        metrics (Dict[str, Any]): A dictionary where keys are metric names (str)
                                  and values are the corresponding metric values.
                                  Values are formatted to 4 decimal places.
        step (int): The current training or evaluation step/iteration.
    """
    # Format metrics for clean output
    formatted_metrics = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"Step {step}: {formatted_metrics}")

# Example usage (optional, for demonstration if this were a standalone script)
if __name__ == "__main__":
    print("This is a utility module. Functions are meant to be imported and used.")
    print("Example of how you might use these functions:")

    # Dummy model and optimizer for demonstration
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyModel()
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)

    # Simulate some training steps
    dummy_input = torch.randn(5, 10)
    dummy_target = torch.randn(5, 1)
    criterion = torch.nn.MSELoss()

    # Create a dummy checkpoint directory
    checkpoint_dir = "temp_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "dummy_checkpoint.pt")

    print("\n--- Simulating Training ---")
    for epoch in range(1, 3):
        dummy_optimizer.zero_grad()
        output = dummy_model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()

        # Simulate exploding gradients for clipping demo
        if epoch == 1:
            for param in dummy_model.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(100.0) # Artificially inflate gradients

        # Clip gradients
        grad_norm = clip_gradients(dummy_model, max_norm=1.0)
        dummy_optimizer.step()

        metrics = {"loss": loss.item(), "grad_norm": grad_norm}
        log_metrics(metrics, step=epoch)

        # Save checkpoint
        save_checkpoint(dummy_model, dummy_optimizer, epoch, checkpoint_path)

    print("\n--- Loading Checkpoint ---")
    try:
        loaded_epoch = load_checkpoint(dummy_model, dummy_optimizer, checkpoint_path)
        print(f"Successfully loaded model, optimizer, and resumed from epoch {loaded_epoch}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during loading: {e}")

    # Clean up dummy checkpoint directory
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(checkpoint_dir):
        os.rmdir(checkpoint_dir)
    print("\nCleaned up dummy checkpoint files.")
