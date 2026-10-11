"""
Model loading utilities with quantization support.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# Prefix for explicit HuggingFace Hub URIs. Bare HF IDs (e.g. "meta-llama/Meta-Llama-3-8B")
# also work and are passed straight through to `from_pretrained`.
HF_URI_PREFIX = "huggingface://"


def download_model_from_huggingface(model_name: str) -> str:
    """
    Resolve a model identifier to a local path via HuggingFace Hub.

    Two forms are accepted:
      1. ``huggingface://<org>/<name>``  -> explicitly pre-downloaded with
         ``huggingface_hub.snapshot_download`` and the local cache path is returned.
      2. Bare HF ID (``<org>/<name>``) or a local filesystem path -> returned as-is;
         ``transformers`` will resolve/download it lazily when ``from_pretrained`` runs.

    Examples of valid ``model_name`` values:
      - "meta-llama/Meta-Llama-3-8B"              (native HF ID, gated: needs --access_token)
      - "mistralai/Mistral-7B-Instruct-v0.2"      (alternative 7B instruct model)
      - "huggingface://meta-llama/Meta-Llama-3-8B" (force pre-download)
      - "./local_model_dir"                        (already-downloaded local path)

    Args:
        model_name: Model identifier, optionally prefixed with ``huggingface://``.

    Returns:
        Local path to the downloaded model, or ``model_name`` unchanged when no
        explicit pre-download is requested.
    """
    if model_name.startswith(HF_URI_PREFIX):
        from huggingface_hub import snapshot_download
        print("Downloading model from HuggingFace Hub...")
        hf_model_name = model_name.removeprefix(HF_URI_PREFIX)
        model_path = snapshot_download(repo_id=hf_model_name)
        print(f"Model downloaded to: {model_path}")
        return model_path
    return model_name


def create_quantization_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float16,
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytesConfig for quantization.
    
    Args:
        load_in_4bit: Whether to use 4-bit quantization
        quant_type: Quantization type ('nf4' or 'fp4')
        compute_dtype: Compute data type
    
    Returns:
        BitsAndBytesConfig instance
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_model(
    model_name: str,
    access_token: str | None = None,
    use_4bit: bool = True,
    trust_remote_code: bool = True,
    gpu_id: int = 0,
):
    """
    Load tokenizer and model with optional quantization.
    
    Args:
        model_name: Name or path of the model
        access_token: HuggingFace access token
        use_4bit: Whether to use 4-bit quantization
        trust_remote_code: Whether to trust remote code
        gpu_id: GPU device ID to use
    
    Returns:
        Tuple of (tokenizer, model)
    """
    print("Loading tokenizer and model...")
    
    # Handle optional huggingface:// URI (bare HF IDs pass through unchanged)
    model_path = download_model_from_huggingface(model_name)
    
    # Create quantization config
    quantization_config = None
    if use_4bit:
        quantization_config = create_quantization_config()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=access_token,
        trust_remote_code=trust_remote_code,
        padding_side="left",
        use_fast=True,
    )
    
    # Handle pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token = eos_token")
    
    # Load model with specific device
    import os
    device = f"cuda:{gpu_id}"
    
    # If CUDA_VISIBLE_DEVICES is set, handle properly
    # Note: When CUDA_VISIBLE_DEVICES restricts visibility, the visible GPUs
    # are re-indexed from 0. So if CUDA_VISIBLE_DEVICES=3, PyTorch sees it as cuda:0
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
        # Check if the requested gpu_id is in the visible list
        visible_gpus = [g.strip() for g in cuda_visible.split(",")]
        if str(gpu_id) in visible_gpus:
            # Find the index of gpu_id in visible list, use that index
            idx = visible_gpus.index(str(gpu_id))
            device = f"cuda:{idx}"
            print(f"CUDA_VISIBLE_DEVICES={cuda_visible}, GPU {gpu_id} is at index {idx}, loading to {device}")
        elif len(visible_gpus) == 1:
            # Only one GPU visible, use cuda:0
            device = "cuda:0"
            print(f"CUDA_VISIBLE_DEVICES={cuda_visible}, single GPU visible, loading to cuda:0")
        else:
            # gpu_id not in visible list, use first visible GPU
            device = "cuda:0"
            print(f"Warning: GPU {gpu_id} not in CUDA_VISIBLE_DEVICES={cuda_visible}, using cuda:0 instead")
    else:
        # No CUDA_VISIBLE_DEVICES set, use gpu_id directly
        print(f"Loading model to {device}")
    
    # Use device_map with single device to ensure proper placement
    device_map = device
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=access_token,
        device_map=device_map,  # Use specific GPU
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
    )
    
    # Sync config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    print(f"Final vocab size: {len(tokenizer)}")
    print(f"pad_token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"eos_token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    
    return tokenizer, model
