"""
Utilities for setting up training
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.distributed as dist

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


def get_smol_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, torch_dtype="bfloat16"
    )


def get_smol_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def set_seed(seed: int = 42) -> None:
    """
    Sets random seed for reproducibility in distributed training

    Args:
        seed: Random seed value, default 42
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class cache_mesh:
    def __init__(self, func):
        self.func = func
        self._mesh = None

    def __call__(self, str, dm: dist.device_mesh.DeviceMesh = None):
        mesh = self._mesh if dm is None else dm
        return self.func(str, mesh)

    def register_mesh(self, mesh: dist.device_mesh.DeviceMesh):
        self._mesh = mesh
        return self


@cache_mesh
def get(str, dm: dist.device_mesh.DeviceMesh = None):
    """
    Applies a func to get whatever is requested.

    `ws` -> dist.get_world_size(pg)
    `pg` -> dist.get_process_group()
    `rank` -> dist.get_rank(pg) # global
    `grank` -> dist.get_rank(pg) # global
    `lrank` -> local_rank
    """

    pg = dm.get_group() if dm else None

    match str:
        case "ws":
            return dist.get_world_size(pg)
        case "pg":
            return pg
        case "rank" | "grank":
            return dist.get_rank(pg)
        case "lrank":
            return dm.get_local_rank() if dm else int(os.environ.get("LOCAL_RANK", 0))
        case _:
            raise ValueError(f"Invalid string: {str}")


def get_dataset():
    dataset = load_dataset("glue", "mrpc")
    tokenizer = get_smol_tokenizer()

    def tokenize_func(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            max_length=None,
            truncation=True,
            padding=True,
        )

    dataset = dataset.map(
        tokenize_func, batched=True, remove_columns=["idx", "sentence1", "sentence2"]
    )
    dataset = dataset.rename_columns({"label": "labels"})
    return dataset


def is_namedtuple(data):
    """
    Checks if `data` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    return (
        isinstance(data, tuple)
        and hasattr(data, "_asdict")
        and hasattr(data, "_fields")
    )


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def recursively_apply(
    func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs
):
    """Recursively applies a function to nested data structures containing tensors."""
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func,
                    o,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for o in data
            ),
        )

    if isinstance(data, dict):
        return {
            k: recursively_apply(
                func,
                v,
                *args,
                test_type=test_type,
                error_on_other_type=error_on_other_type,
                **kwargs,
            )
            for k, v in data.items()
        }

    if test_type(data):
        return func(data, *args, **kwargs)

    if error_on_other_type:
        raise TypeError(
            f"Unsupported type {type(data)} passed to {func.__name__}. "
            f"Only nested containers of {test_type.__name__} objects allowed."
        )

    return data


def gather(tensor: torch.Tensor, pg: dist.ProcessGroup = None):
    def _gather_one(tensor):
        # Single dim tensor, just copy it
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        # We can only gather contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Create a tensor of correct ending size
        output_tensors = torch.empty(
            tensor.numel() * get("ws", pg), dtype=tensor.dtype, device=tensor.device
        )
        dist.all_gather(output_tensors, tensor, group=pg)
        return output_tensors.view(-1, *tensor.size()[1:])

    return recursively_apply(_gather_one, tensor, pg=pg, error_on_other_type=True)
