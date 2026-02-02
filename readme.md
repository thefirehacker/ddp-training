# Distributed Data Parallel Training on Modal - Multi GPU

This project demonstrates PyTorch Distributed Data Parallel (DDP) training running on Modal's cloud GPUs, including NanoGPT speedrun implementations.

## Prerequisites

1. Install the Modal CLI:
   ```bash
   pip install modal
   ```

2. Authenticate with Modal:
   ```bash
   modal token new
   ```

3. Create a wandb secret for logging (required for speedruns):
   ```bash
   modal secret create wandb-secret WANDB_API_KEY=your_api_key_here
   ```
   
   Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).

---

## Project 1: Basic DDP Training

Run basic DDP training on 2x L40S GPUs:

```bash
modal run modal_ddp.py::train_single_node
```

This will:
- Launch a container with 2x L40S GPUs on Modal
- Run `ddp.py` using `torchrun` for distributed training
- Save profiler traces to a persistent Modal Volume
- Log training metrics to Weights & Biases

---

## Project 2: NanoGPT Speedrun (Tyler Romero)

Progressive GPT-2 training optimization following [Tyler Romero's worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/). Run each step to see how optimizations improve training time.

### Steps Overview

| Step | Folder | Expected Time | Key Changes |
|------|--------|---------------|-------------|
| 1 | 01-Initialbaseline | ~8 hrs | Base GPT-2 from llm.c |
| 2 | 02-ArchitecturalChanges | ~7.5 hrs | RoPE, ReLU², trapezoidal LR |
| 3 | 03-MuonOptimizer | ~4.5 hrs | Muon optimizer |
| 4 | 04-DataLoadingTwerks | ~4.3 hrs | Micro-batch loading |
| 5 | 05-LogitSoftCappingat30 | ~4 hrs | Logit soft-capping |
| 6 | 06-LongerSequenceLength | ~2.5 hrs | FlexAttention, 32K seq |

### Running Tyler's Speedrun

```bash
# First time: download data (900M tokens, ~5min)
modal run nanogpt-speedrun/src/runfiles/modal_runner.py::download_data

# Run any step (1-6)
modal run nanogpt-speedrun/src/runfiles/modal_runner.py::train --step 1
modal run nanogpt-speedrun/src/runfiles/modal_runner.py::train --step 2
modal run nanogpt-speedrun/src/runfiles/modal_runner.py::train --step 3
modal run nanogpt-speedrun/src/runfiles/modal_runner.py::train --step 4
modal run nanogpt-speedrun/src/runfiles/modal_runner.py::train --step 5
modal run nanogpt-speedrun/src/runfiles/modal_runner.py::train --step 6

# With custom notes
modal run nanogpt-speedrun/src/runfiles/modal_runner.py::train --step 1 --notes "first attempt"
```

**Wandb Project**: `tyler-nanogpt-run`

---

## Project 3: Modded-NanoGPT (Keller Jordan)

World-record NanoGPT speedrun by Keller Jordan. Trains GPT-2 to 3.28 val loss in under 100 seconds on 8x H100.

### Running Keller's Speedrun

```bash
# First time: download data
modal run modded-nanogpt/modal_modded_nanogpt.py::download_data

# Run the speedrun (requires 8x H100)
modal run modded-nanogpt/modal_modded_nanogpt.py::train
```

**Note**: torch.compile adds ~7 minutes latency on first run.

**Wandb Project**: `modded-nanogpt-run`

---

## Viewing Training Logs in Wandb

After training, view your logs at [wandb.ai](https://wandb.ai):
- `ddp-training` - Basic DDP training
- `tyler-nanogpt-run` - Tyler Romero's speedrun steps
- `modded-nanogpt-run` - Keller Jordan's world-record speedrun

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
   - Gradient synchronization
   - Optimizer step timing

---

## Project Structure

```
├── ddp.py                          # Basic DDP training script
├── modal_ddp.py                    # Modal wrapper for DDP
├── training_utils/                 # Shared utilities
│   ├── __init__.py
│   ├── memory.py
│   ├── trun.py
│   └── utils.py
├── nanogpt-speedrun/               # Tyler Romero's speedrun
│   ├── src/runfiles/
│   │   ├── modal_runner.py         # Modal wrapper
│   │   ├── 01-Initialbaseline/
│   │   ├── 02-ArchitecturalChanges/
│   │   ├── 03-MuonOptimizer/
│   │   ├── 04-DataLoadingTwerks/
│   │   ├── 05-LogitSoftCappingat30/
│   │   └── 06-LongerSequenceLength/
│   └── pyproject.toml
├── modded-nanogpt/                 # Keller Jordan's speedrun
│   ├── modal_modded_nanogpt.py     # Modal wrapper
│   ├── train_gpt.py
│   ├── Dockerfile
│   └── requirements.txt
└── readme.md
```

## GPU Configuration

| Project | Default GPU | Cost/hr |
|---------|-------------|---------|
| DDP Training | L40S:2 | ~$2.60 |
| Tyler's Speedrun | L40S:2 | ~$2.60 |
| Keller's Speedrun | H100:8 | ~$32 |

Edit the Modal wrapper files to change GPU type/count.
