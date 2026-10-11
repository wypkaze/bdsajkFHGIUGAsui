"""
Model-related utility functions.
"""

import torch


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model to inspect
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )
