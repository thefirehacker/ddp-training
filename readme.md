# LLM Training: PyTorch DDP → NanoGPT Speedrun

Learn how an LLM training run works from the actual code — starting with a small PyTorch Distributed Data Parallel example and progressing to NanoGPT speedruns, Muon, H100 profiling, and Keller Jordan's modded-nanogpt.

This repository is the runnable-code companion to **[FBA Lab](https://bubblnet.com/lab/speedrun)**.

**Want to understand the code before running it?**

→ [Open the interactive LLM Training Speedrun](https://bubblnet.com/lab/speedrun)

FBA Lab keeps the **training code, model architecture, explanations, visualizations, and real training-run data** together while you work through the run.

---

## What's in this repo?

There are three useful starting points.

| Project | What it teaches | Hardware |
|---|---|---|
| **PyTorch DDP** | How multiple GPUs train synchronized copies of one model | 2× L40S |
| **NanoGPT Speedrun** | Progressive GPT-2 training optimizations from Tyler Romero's speedrun | 8× H100 |
| **modded-nanogpt** | Keller Jordan's highly optimized NanoGPT training stack | 8× H100 |

If you are learning distributed training for the first time, start with **Project 1**.

If you already understand basic PyTorch training and want to study LLM training systems, start with **Project 2**.

---

# 1. PyTorch Distributed Data Parallel

The simplest project in the repository.

Start here if you want to understand what happens when one PyTorch training job is spread across multiple GPUs.

Files:

- [`ddp.py`](ddp.py) — learner-friendly DDP training script
- [`modal_ddp.py`](modal_ddp.py) — launches the job on Modal
- [`training_utils/`](training_utils/) — supporting utilities

The important ideas are:

```text
different batch on each GPU
        ↓
same model on each GPU
        ↓
forward pass
        ↓
loss.backward()
        ↓
gradient synchronization
        ↓
optimizer.step()
        ↓
identical updated model on every GPU
```

Run it on 2× L40S GPUs:

```bash
modal run modal_ddp.py::train_single_node
```

This will:

- launch a 2-GPU Modal container
- start the processes with `torchrun`
- train using PyTorch DDP
- capture profiler traces
- log training metrics to Weights & Biases

---

# 2. NanoGPT Speedrun

This section follows the progressive GPT-2 training optimizations documented by Tyler Romero.

Instead of jumping directly into a highly optimized training script, the run is broken into a sequence of changes.

Directory:

```text
nanogpt-speedrun/
```

Modal entrypoint:

```text
nanogpt-speedrun/src/runfiles/modal_runner.py
```

## The six steps

| Step | Experiment | Main change |
|---|---|---|
| 1 | Initial baseline | Simple GPT-2 training run |
| 2 | Architecture changes | RoPE, ReLU², learning-rate changes |
| 3 | Muon optimizer | Introduces Muon |
| 4 | Data loading | Improves the training input pipeline |
| 5 | Logit soft-capping | Caps extreme logits |
| 6 | Longer sequence length | Longer context and attention changes |

The corresponding source folders are:

```text
nanogpt-speedrun/src/runfiles/
├── 01-Initialbaseline/
├── 02-ArchitecturalChanges/
├── 03-MuonOptimizer/
├── 04-DataLoadingTwerks/
├── 05-LogitSoftCappingat30/
└── 06-LongerSequenceLength/
```

### Study the run interactively first

The FBA Lab version lets you move through the same progression while keeping the code, architecture, explanation and run data synchronized:

**https://bubblnet.com/lab/speedrun**

Start directly with the baseline:

**https://bubblnet.com/lab/speedrun/journey/tyler-01**

---

## Running the NanoGPT speedrun on Modal

First install Modal:

```bash
pip install modal
```

Authenticate:

```bash
modal token new
```

Create the required secrets:

```bash
modal secret create wandb-secret WANDB_API_KEY=your_api_key_here
modal secret create HF_TOKEN HF_TOKEN=your_huggingface_token_here
```

The speedrun uses:

```text
GPUs:        H100:8
Data volume: fineweb-data
Trace volume: ddp-traces
W&B secret:  wandb-secret
HF secret:   HF_TOKEN
```

If the FineWeb data is not already present:

```bash
cd nanogpt-speedrun

modal run src/runfiles/modal_runner.py::download_data
```

Run the baseline:

```bash
modal run src/runfiles/modal_runner.py::train --step 1
```

Then run the successive experiments:

```bash
modal run src/runfiles/modal_runner.py::train --step 2
modal run src/runfiles/modal_runner.py::train --step 3
modal run src/runfiles/modal_runner.py::train --step 4
modal run src/runfiles/modal_runner.py::train --step 5
modal run src/runfiles/modal_runner.py::train --step 6
```

You can attach notes to a run:

```bash
modal run src/runfiles/modal_runner.py::train \
  --step 1 \
  --notes "first attempt"
```

---

## What the baseline runner does

For Step 1, the Modal runner:

1. mounts the FineWeb data
2. mounts the profiler-trace volume
3. selects the baseline `train_gpt2.py`
4. starts the distributed run with `torchrun`
5. launches the job across 8 GPUs
6. logs metrics to Weights & Biases

The baseline walkthrough is also available here:

```text
tutorial/tyler-speedrun/Whatisbaseline.md
```

Interactive version:

**https://bubblnet.com/lab/speedrun/journey/tyler-01**

---

# 3. Keller Jordan's modded-nanogpt

The `modded-nanogpt/` directory contains the path for running Keller Jordan's highly optimized NanoGPT training stack on Modal.

Directory:

```text
modded-nanogpt/
```

Enter the directory:

```bash
cd modded-nanogpt
```

Download the data if needed:

```bash
modal run modal_modded_nanogpt.py::download_data
```

Run training:

```bash
modal run modal_modded_nanogpt.py::train
```

Run with PyTorch profiler capture:

```bash
modal run modal_modded_nanogpt.py::train --profiler
```

The standard run logs scalar metrics to W&B.

The profiler-enabled run additionally captures profiler traces.

---

# H100 profiler traces

One of the most useful ways to understand a real training run is to look at where GPU time actually goes.

Profiler traces can show:

- forward-pass kernels
- backward-pass kernels
- gradient synchronization
- NCCL communication
- memory operations
- optimizer work
- kernel launches
- CPU/GPU gaps

After a profiler-enabled run:

```bash
modal volume ls ddp-traces
```

Download the traces:

```bash
modal volume get ddp-traces / ./local_traces
```

Open the resulting `.pt.trace.json` file in Perfetto:

```text
https://ui.perfetto.dev
```

FBA Lab also walks through the training system interactively:

**https://bubblnet.com/lab/speedrun**

---

# Weights & Biases

The projects used by the repository are:

```text
ddp-training
tyler-nanogpt-run
modded-nanogpt-run
```

The normal training runs log scalar metrics such as loss and training time.

Profiler artifacts are generated only when profiling is enabled.

---

# Repository structure

```text
.
├── ddp.py
├── modal_ddp.py
│
├── training_utils/
│
├── nanogpt-speedrun/
│   └── src/
│       └── runfiles/
│           ├── 01-Initialbaseline/
│           ├── 02-ArchitecturalChanges/
│           ├── 03-MuonOptimizer/
│           ├── 04-DataLoadingTwerks/
│           ├── 05-LogitSoftCappingat30/
│           └── 06-LongerSequenceLength/
│
├── modded-nanogpt/
│
└── tutorial/
```

---

# Suggested learning path

If you're here to **learn**, use this order:

```text
1. Read ddp.py
       ↓
2. Understand ranks and DDP
       ↓
3. Follow the NanoGPT baseline
       ↓
4. Trace forward → loss → backward → optimizer
       ↓
5. Study the six speedrun iterations
       ↓
6. Inspect the H100 profiler
       ↓
7. Read the modded-nanogpt optimizations
```

The interactive version of this path is in FBA Lab:

### [Open the LLM Training Speedrun →](https://bubblnet.com/lab/speedrun)

No GPU is required to study the interactive run.

---

# Upstream work

This repo includes and builds on:

- [tyler-romero/nanogpt-speedrun](https://github.com/tyler-romero/nanogpt-speedrun)
- [kellerjordan/modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt)

Original MIT licenses remain in [`nanogpt-speedrun/LICENSE`](nanogpt-speedrun/LICENSE) and [`modded-nanogpt/LICENSE`](modded-nanogpt/LICENSE).

The goal of this repository is educational: make the training code runnable, inspectable and easier to understand.

---

## FBA Lab

**You know the pieces. See the whole AI model training run.**

FBA Lab is an interactive environment for studying real AI model training through synchronized:

```text
CODE ↔ ARCHITECTURE ↔ EXPLANATION ↔ TRAINING RUN
```

Start here:

**https://bubblnet.com/lab/speedrun**
