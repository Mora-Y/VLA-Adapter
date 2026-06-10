"""Utilities for saving trainable VLA checkpoint components."""

from pathlib import Path
from typing import Any, Dict, Union

import torch

ACTION_QUERIES_KEY = "action_queries.weight"
ACTION_QUERIES_MODULE_NAME = "action_queries"
TRAINING_STATE_MODULE_NAME = "training_state"


def action_queries_checkpoint_path(checkpoint_dir: Union[str, Path], checkpoint_name_suffix: str) -> Path:
    return Path(checkpoint_dir) / f"{ACTION_QUERIES_MODULE_NAME}--{checkpoint_name_suffix}"


def training_state_checkpoint_path(checkpoint_dir: Union[str, Path], checkpoint_name_suffix: str) -> Path:
    return Path(checkpoint_dir) / f"{TRAINING_STATE_MODULE_NAME}--{checkpoint_name_suffix}"


def extract_action_queries_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a portable state dict for the trainable action query embeddings."""
    for name, tensor in model.state_dict().items():
        if name.endswith(ACTION_QUERIES_KEY):
            return {ACTION_QUERIES_KEY: tensor.detach().cpu()}
    raise KeyError(f"Could not find `{ACTION_QUERIES_KEY}` in model state dict")


def load_action_queries_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load action query embeddings into wrapped or unwrapped VLA modules."""
    if ACTION_QUERIES_KEY not in state_dict:
        raise KeyError(f"Missing `{ACTION_QUERIES_KEY}` in action query checkpoint")

    target_key = None
    model_state = model.state_dict()
    for name in model_state:
        if name.endswith(ACTION_QUERIES_KEY):
            target_key = name
            break

    if target_key is None:
        raise KeyError(f"Could not find `{ACTION_QUERIES_KEY}` in target model")

    target = model_state[target_key]
    source = state_dict[ACTION_QUERIES_KEY].to(device=target.device, dtype=target.dtype)
    if source.shape != target.shape:
        raise ValueError(
            f"Action query shape mismatch: checkpoint has {tuple(source.shape)}, "
            f"model expects {tuple(target.shape)}"
        )

    with torch.no_grad():
        model_state[target_key].copy_(source)


def save_action_queries_checkpoint(
    model: torch.nn.Module, checkpoint_dir: Union[str, Path], checkpoint_name_suffix: str
) -> None:
    torch.save(
        extract_action_queries_state_dict(model),
        action_queries_checkpoint_path(checkpoint_dir, checkpoint_name_suffix),
    )


def save_training_state_checkpoint(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_dir: Union[str, Path],
    checkpoint_name_suffix: str,
    step: int,
) -> None:
    torch.save(
        {
            "step": int(step),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        training_state_checkpoint_path(checkpoint_dir, checkpoint_name_suffix),
    )


def load_training_state_checkpoint(
    checkpoint_dir: Union[str, Path], checkpoint_name_suffix: str, map_location: Any = "cpu"
) -> Dict[str, Any]:
    return torch.load(
        training_state_checkpoint_path(checkpoint_dir, checkpoint_name_suffix),
        map_location=map_location,
        weights_only=False,
    )
