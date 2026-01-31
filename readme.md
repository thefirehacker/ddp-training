# Distributed Data Parallel Training on Modal - Multi GPU

This project demonstrates PyTorch Distributed Data Parallel (DDP) training running on Modal's cloud GPUs.

## Prerequisites

1. Install the Modal CLI:
   ```bash
   pip install modal
   ```

2. Authenticate with Modal:
   ```bash
   modal token new
   ```

3. Create a wandb secret for logging (optional but recommended):
   ```bash
   modal secret create wandb-secret WANDB_API_KEY=your_api_key_here
   ```
   
   Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).

## Running DDP Training

Run the training on 2x L40S GPUs:

```bash
modal run modal_ddp.py::train_single_node
```

This will:
- Launch a container with 2x L40S GPUs on Modal
- Run `ddp.py` using `torchrun` for distributed training
- Save profiler traces to a persistent Modal Volume
- Log training metrics to Weights & Biases (if wandb secret is configured)

## Viewing Training Logs in Wandb

After training, view your logs at [wandb.ai](https://wandb.ai). The run will appear under the `ddp-training` project with metrics including:
- Loss per step
- Training configuration (model, batch size, GPU count)

## Downloading Profiler Traces

After training completes, download the traces:

```bash
# List available traces
modal volume ls ddp-traces

# Download all traces to a local directory
modal volume get ddp-traces / ./local_traces
```

## Viewing Traces in Perfetto

1. Open [ui.perfetto.dev](https://ui.perfetto.dev) in your browser
2. Click "Open trace file" or drag and drop
3. Select the `.pt.trace.json` file from your downloaded traces
4. Explore the timeline to analyze:
   - Forward/backward pass timing
   - Data movement overhead
   - Gradient synchronization (`sync_grads`)
   - Optimizer step timing

## Project Structure

```
├── ddp.py              # DDP training script with profiler
├── modal_ddp.py        # Modal wrapper for cloud execution
├── training_utils/
│   ├── __init__.py
│   ├── memory.py       # Memory tracking utilities
│   ├── trun.py         # Torchrun wrapper
│   └── utils.py        # Model, dataset, and distributed utilities
└── readme.md
```

## GPU Configuration

Edit `modal_ddp.py` to change GPU type/count:

```python
# Examples:
@app.function(gpu="L40S:2", ...)   # 2x L40S (current)
@app.function(gpu="L40S:4", ...)   # 4x L40S
@app.function(gpu="A100:2", ...)   # 2x A100
@app.function(gpu="H100:2", ...)   # 2x H100
```

Remember to update `--nproc_per_node` in the `torchrun` command to match the GPU count.
