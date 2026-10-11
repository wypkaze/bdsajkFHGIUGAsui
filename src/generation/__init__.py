"""Generation utilities (thin batch layer over pair_generator + clean_output)."""

from .generator import (
    AnchorPromptBuilder,
    generate_batch,
    batch_generate_raw,
    clean_generation,
)

__all__ = [
    "AnchorPromptBuilder",
    "generate_batch",
    "batch_generate_raw",
    "clean_generation",
]
