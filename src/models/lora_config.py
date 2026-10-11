"""
LoRA configuration and application utilities.
"""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 8,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
) -> LoraConfig:
    """
    Create LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        target_modules: List of module names to apply LoRA
        lora_dropout: Dropout rate
        bias: Bias training mode
    
    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )


def apply_lora_to_model(
    model: torch.nn.Module,
    lora_config: LoraConfig | None = None,
    prepare_for_kbit: bool = True,
) -> torch.nn.Module:
    """
    Apply LoRA to a model.
    
    Args:
        model: Base model to apply LoRA
        lora_config: LoRA configuration (creates default if None)
        prepare_for_kbit: Whether to prepare model for k-bit training
    
    Returns:
        Model with LoRA applied
    """
    # Prepare for k-bit training if needed
    if prepare_for_kbit:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    
    # Create default config if not provided
    if lora_config is None:
        lora_config = create_lora_config()
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model = model.to("cuda")
    
    return model
