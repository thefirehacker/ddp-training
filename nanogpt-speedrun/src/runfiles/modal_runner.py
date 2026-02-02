import modal
from pathlib import Path

# Volumes - reuse existing ddp-traces volume
traces = modal.Volume.from_name("ddp-traces", create_if_missing=True)
data_vol = modal.Volume.from_name("fineweb-data", create_if_missing=True)
TRACE_DIR = Path("/traces")
DATA_DIR = Path("/data")

WANDB_PROJECT = "tyler-nanogpt-run"

# Step folder mapping
STEPS = {
    1: "01-Initialbaseline",
    2: "02-ArchitecturalChanges",
    3: "03-MuonOptimizer",
    4: "04-DataLoadingTwerks",
    5: "05-LogitSoftCappingat30",
    6: "06-LongerSequenceLength",
}

# Image with uv installed
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .add_local_dir(".", remote_path="/root/nanogpt-speedrun")
    .workdir("/root/nanogpt-speedrun")
    .run_commands("uv sync --all-extras")
)

app = modal.App("tyler-nanogpt-speedrun", image=image)


@app.function(
    gpu="L40S:2",
    timeout=2 * 60 * 60,
    volumes={DATA_DIR: data_vol},
)
def download_data(num_chunks: int = 9):
    """
    Download FineWeb10B data to persistent volume.
    
    Args:
        num_chunks: Number of 100M token chunks to download (default 9 = 900M tokens)
    """
    import subprocess
    
    print(f"Downloading {num_chunks} chunks (~{num_chunks * 100}M tokens)...")
    
    subprocess.run(
        ["uv", "run", "python", "data/cached_fineweb10B.py", str(num_chunks)],
        cwd="/root/nanogpt-speedrun",
        check=True,
    )
    data_vol.commit()
    print(f"Data downloaded and saved to Modal Volume 'fineweb-data'")


@app.function(
    gpu="L40S:2",
    timeout=10 * 60 * 60,  # 10 hours max (for baseline)
    volumes={TRACE_DIR: traces, DATA_DIR: data_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(step: int = 1, notes: str = ""):
    """
    Run GPT-2 speedrun training for specified step.
    
    Steps:
        1: Initial baseline (~8 hrs)
        2: Architectural changes (~7.5 hrs)
        3: Muon optimizer (~4.5 hrs)
        4: Dataloading tweaks (~4.3 hrs)
        5: Logit soft-capping (~4 hrs)
        6: Longer sequence length (~2.5 hrs)
    
    Args:
        step: Step number (1-6)
        notes: Optional notes for wandb run
    """
    import subprocess
    import sys
    import os
    import shutil

    if step not in STEPS:
        raise ValueError(f"Invalid step {step}. Must be 1-6. Available steps: {list(STEPS.keys())}")
    
    step_folder = STEPS[step]
    step_path = f"src/runfiles/{step_folder}"
    train_file = f"{step_path}/train_gpt2.py"
    run_script = f"{step_path}/run.sh"
    
    env = os.environ.copy()
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["TRACE_DIR"] = str(TRACE_DIR)
    
    run_notes = f"{step_folder}: {notes}" if notes else step_folder

    print("=" * 60)
    print(f"Tyler Romero NanoGPT Speedrun - Step {step}")
    print("=" * 60)
    print(f"Step folder: {step_folder}")
    print(f"Train file: {train_file}")
    print(f"Run script: {run_script}")
    print(f"Wandb Project: {WANDB_PROJECT}")
    print(f"Notes: {run_notes}")
    print("=" * 60)

    # Copy step's train_gpt2.py to src/train_gpt2.py (where run.sh expects it)
    print(f"Copying {train_file} -> src/train_gpt2.py")
    shutil.copy(train_file, "src/train_gpt2.py")

    # Execute the step's run.sh with notes
    print(f"Executing: bash {run_script} '{run_notes}'")
    result = subprocess.run(
        ["bash", run_script, run_notes],
        cwd="/root/nanogpt-speedrun",
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    # Commit traces to persistent volume
    traces.commit()
    print(f"\nTraces saved to Modal Volume 'ddp-traces'")
    print(f"Download with: modal volume get ddp-traces / ./local_traces")
    
    if result.returncode != 0:
        print(f"Training failed with exit code: {result.returncode}")
    else:
        print("Training completed successfully!")
    
    return result.returncode
