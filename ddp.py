import os

import torch
import torch.distributed as dist
import wandb

from accelerate import PartialState
from torch.utils.data import DataLoader

from training_utils.utils import get_smol_model, get_dataset, get, get_smol_tokenizer, set_seed

from torch.profiler import profile, record_function, ProfilerActivity, schedule


class SimpleDistributedDataParallelism:
    def __init__(self, model: torch.nn.Module):
        self.model = model

        for param in model.parameters():
            rank0_param = param.data.clone()
            dist.broadcast(rank0_param, src=0)
            if not torch.equal(param.data, rank0_param):
                raise ValueError(
                    "Expected model parameters to be identical during `__init__`, but this is not true. "
                    "Make sure to set the seeds before creating your model"
                )

    def sync_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


if __name__ == "__main__":
    state = PartialState()
    device = state.device

    # Initialize wandb on main process only
    if state.is_main_process:
        wandb.init(
            project="ddp-training",
            config={
                "model": "SmolLM2-360M",
                "batch_size": 8,
                "num_gpus": dist.get_world_size() if dist.is_initialized() else 1,
                "optimizer": "SGD",
                "learning_rate": 1e-3,
            }
        )

    set_seed(42)

    dataset = get_dataset()["train"]
    train_ds = dataset.shuffle(seed=42)

    tokenizer = get_smol_tokenizer()

    def collate_func(batch):
        return tokenizer.pad(
            batch,
            padding="longest",
            max_length=None,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    per_device_batch_size = 16

    train_dataloader = DataLoader(
        train_ds,
        batch_size=per_device_batch_size,
        collate_fn=collate_func,
        drop_last=True,
        shuffle=True
    )

    model = get_smol_model()
    model.to(device)
    optimizer = torch.optim.SGD(model.model.parameters(), lr=1e-3)

    train_ds = dataset.shuffle(seed=42)

    def collate_func(batch):
        return tokenizer.pad(
            batch,
            padding="longest",
            max_length=None,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    per_device_batch_size = 8

    # do initial shuffle
    train_ds = dataset.shuffle(seed=42)

    # Shard data for first parallel dimension
    # Takes dataset of [0, 1, 2, ... n] -> [[0, 1, 2, ... n/ws], [n/ws, n/ws+1, ... n-1]]
    ds_length = len(train_ds)
    ds_length_per_rank = ds_length // get("ws")
    rank = get("rank")
    start = rank * ds_length_per_rank
    end = start + ds_length_per_rank if rank != get("ws") - 1 else ds_length

    train_shard = train_ds.select(list(range(start, end)))

    train_dataloader = DataLoader(
        train_shard,
        batch_size=per_device_batch_size,
        collate_fn=collate_func,
        drop_last=True,
        shuffle=True
    )

    model = get_smol_model()
    model.to(device)
    optimizer = torch.optim.SGD(model.model.parameters(), lr=1e-3)
    model = SimpleDistributedDataParallelism(model)

    profiler_context = None

    if state.is_main_process:
        profiler_schedule = schedule(
            skip_first=5,
            wait=1,
            warmup=2,
            active=5,
            repeat=1
        )

        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.environ.get("TRACE_DIR", "ddp_trace")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )

    if profiler_context:
        profiler_context.__enter__()

    num_batches = 0
    for (i, batch) in enumerate(train_dataloader):
        if i > 20:
            break
        # Move batch to GPU
        with record_function("data_movement"):
            batch = {k: v.to(device) for k, v in batch.items()}

        with record_function("forward"):
            output = model(**batch)
        with record_function("backward"):
            output.loss.backward()

        with record_function("sync_grads"):
            model.sync_gradients()

        with record_function("opt_step"):
            optimizer.step()
            optimizer.zero_grad()
            if profiler_context:
                profiler_context.step()

        # Log metrics to wandb
        if state.is_main_process:
            wandb.log({
                "loss": output.loss.item(),
                "step": i,
            })

    if profiler_context:
        profiler_context.__exit__(None, None, None)

    # Finish wandb logging
    if state.is_main_process:
        wandb.finish()

    dist.destroy_process_group()
