# Distributed Data Parallel Training on Modal - Multi GPU

This project demonstrates PyTorch Distributed Data Parallel (DDP) training running on Modal's cloud GPUs, including NanoGPT speedrun implementations.

This repository is part of **[First Break AI](https://cohort.bubblnet.com/)** — a free, open cohort for learning training, inference, and AI product building. The roadmap, checklist, and community live on that site.

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
# Navigate to nanogpt-speedrun folder first
cd nanogpt-speedrun

# First time: download data (900M tokens, ~5min)
modal run src/runfiles/modal_runner.py::download_data

# Run any step (1-6)
modal run src/runfiles/modal_runner.py::train --step 1
modal run src/runfiles/modal_runner.py::train --step 2
modal run src/runfiles/modal_runner.py::train --step 3
modal run src/runfiles/modal_runner.py::train --step 4
modal run src/runfiles/modal_runner.py::train --step 5
modal run src/runfiles/modal_runner.py::train --step 6

# With custom notes
modal run src/runfiles/modal_runner.py::train --step 1 --notes "first attempt"

# Return to root folder
cd ..
```

**Wandb Project**: `tyler-nanogpt-run`

---

## Project 3: Modded-NanoGPT (Keller Jordan)

World-record NanoGPT speedrun by Keller Jordan. Trains GPT-2 to 3.28 val loss in under 100 seconds on 8x H100.

### Running Keller's Speedrun

```bash
# Navigate to modded-nanogpt folder first (required for Dockerfile)
cd modded-nanogpt

# First time: download data
modal run modal_modded_nanogpt.py::download_data

# Run the speedrun (requires 8x H100)
modal run modal_modded_nanogpt.py::train

# Optional: capture PyTorch profiler traces to ddp-traces + W&B Artifacts (adds overhead)
modal run modal_modded_nanogpt.py::train --profiler

# Return to root folder
cd ..
```

**Note**: torch.compile adds ~7 minutes latency on first run.

**Wandb Project**: `modded-nanogpt-run`

**Metrics vs traces**

| Output | Default `train` | `train --profiler` |
|--------|-----------------|---------------------|
| W&B Charts (`val_loss`, `train_time_ms`, …) | Yes | Yes |
| W&B Artifacts (`profiler-traces`) | No | Yes (if traces written) |
| Modal volume `ddp-traces` (`.pt.trace.json`) | No | Yes |

`download_data` does not write traces; only training with the profiler flag does.

---

## Viewing Training Logs in Wandb

After training, view your logs at [wandb.ai](https://wandb.ai):
- `ddp-training` - Basic DDP training
- `tyler-nanogpt-run` - Tyler Romero's speedrun steps
- `modded-nanogpt-run` - Keller Jordan's world-record speedrun

**Modded-nanogpt:** Charts tab shows scalars from every run. The **Artifacts** tab gets `profiler-traces` only when you run `modal run modal_modded_nanogpt.py::train --profiler` (or set `ENABLE_PROFILER=1` in the training env).

## Downloading Profiler Traces

After a **profiler-enabled** modded-nanogpt run (or Project 1 DDP training), download traces from Modal:

```bash
# List available traces
modal volume ls ddp-traces

# Download all traces to a local directory
modal volume get ddp-traces / ./local_traces
```

Project 1 (`modal_ddp.py`) always profiles. Keller's speedrun profiles only with `--profiler`.

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

---

## How Modal Runs These Projects

All three projects run on Modal's cloud infrastructure. When you run `modal run <file>::<function>`, Modal:

1. Parses your Python file
2. Builds (or reuses cached) container image
3. Spins up container with GPUs attached
4. Mounts volumes and injects secrets
5. Runs your function
6. Tears down container when done

```
┌─────────────────────────────────────────────────────────────┐
│  Your laptop                                                │
│  $ modal run modal_modded_nanogpt.py::train                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Modal Cloud                                                │
│                                                             │
│  1. Check if image is cached                                │
│     ├── Cached? → Use existing image                        │
│     └── Not cached? → Build new image                       │
│                                                             │
│  2. Spin up container with:                                 │
│     - GPUs attached (L40S, H100, etc.)                      │
│     - Volume mounts (fineweb-data, ddp-traces)              │
│     - Secrets injected (WANDB_API_KEY)                      │
│                                                             │
│  3. Run function: train()                                   │
│     └── subprocess: torchrun train_gpt.py                   │
│                                                             │
│  4. Container stops when function returns                   │
│     - Volumes persisted (data survives)                     │
│     - Container deleted (ephemeral)                         │
└─────────────────────────────────────────────────────────────┘
```

### Container Image Methods

Modal supports two ways to define container images:

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Image Builder** | Chain of Modal commands | Simple pip dependencies |
| **Dockerfile** | Standard Docker file | Complex builds, custom OS setup |

### How Each Project Builds Its Image

| Project | Method | Why |
|---------|--------|-----|
| Basic DDP | Image Builder | Simple - just pip packages |
| Tyler's Speedrun | Image Builder | Uses `uv` package manager |
| Keller's Modded | Dockerfile | Complex - custom Python 3.12, CUDA 12.6, Flash Attention 3 |

**Basic DDP** (`modal_ddp.py`):
```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch~=2.5.1", "accelerate", ...)
)
```

**Tyler's Speedrun** (`modal_runner.py`):
```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .add_local_dir(".", remote_path="/root/nanogpt-speedrun", copy=True)
    .run_commands("uv sync --all-extras")
)
```

**Keller's Modded** (`modal_modded_nanogpt.py`):
```python
image = modal.Image.from_dockerfile("Dockerfile")
```

### Shared Resources

All projects share these Modal resources:

| Resource | Type | Purpose |
|----------|------|---------|
| `fineweb-data` | Volume | Pre-tokenized training data (1.9 GB) |
| `ddp-traces` | Volume | Profiler traces |
| `wandb-secret` | Secret | Weights & Biases API key |

Data downloaded once is available to all projects.
