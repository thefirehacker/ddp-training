import modal
import os
from pathlib import Path

# Volumes - reuse existing volumes (shared with modded-nanogpt)
traces = modal.Volume.from_name("ddp-traces", create_if_missing=True)
data_vol = modal.Volume.from_name("fineweb-data", create_if_missing=True)
TRACE_DIR = Path("/traces")
# Mount data volume at the path where run.sh expects it
DATA_DIR = Path("/root/nanogpt-speedrun/src/data/fineweb10B")

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
    .add_local_dir(".", remote_path="/root/nanogpt-speedrun", copy=True)  # copy=True allows subsequent commands
    .workdir("/root/nanogpt-speedrun")
    .run_commands("uv sync --all-extras")
)

app = modal.App("tyler-nanogpt-speedrun", image=image)


def _trace_files_in_dir(trace_dir: Path) -> set[str]:
    if not trace_dir.exists():
        return set()
    return {
        name
        for name in os.listdir(trace_dir)
        if name.endswith((".json", ".json.gz")) or ".trace" in name
    }


@app.function(
    timeout=2 * 60 * 60,  # No GPU needed for download
    volumes={DATA_DIR: data_vol},
    secrets=[modal.Secret.from_name("HF_TOKEN")],
)
def download_data(num_chunks: int = 9):
    """
    Download FineWeb10B data to persistent volume.

    Note: This volume is shared with modded-nanogpt. If you already ran
    download_data from either project, the data is already here.

    Args:
        num_chunks: Number of 100M token chunks to download (default 9 = 900M tokens)
    """
    import subprocess
    import sys
    import os

    # Debug: show volume mount
    print(f"Volume mounted at: {DATA_DIR}")
    print(f"Volume exists: {os.path.exists(DATA_DIR)}")
    if os.path.exists(DATA_DIR):
        print(f"Volume contents: {os.listdir(DATA_DIR)}")

    # Check if data already exists (shared from modded-nanogpt or a prior run)
    val_file = str(DATA_DIR / "fineweb_val_000000.bin")
    if os.path.exists(val_file):
        existing_files = len([f for f in os.listdir(DATA_DIR) if f.endswith(".bin")])
        print(f"Data already exists! Found {existing_files} .bin files")
        print("Skipping download. Data was likely downloaded previously.")
        return

    print(f"Downloading {num_chunks} chunks (~{num_chunks * 100}M tokens)...")
    print("This will take several minutes...")

    result = subprocess.run(
        ["uv", "run", "python", "src/data/cached_fineweb10B.py", str(num_chunks)],
        cwd="/root/nanogpt-speedrun",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        print(f"Download failed with exit code: {result.returncode}")
        return

    # Show what was downloaded
    if os.path.exists(DATA_DIR):
        files = os.listdir(DATA_DIR)
        print(f"Downloaded {len(files)} files to volume")
        for f in sorted(files)[:5]:
            print(f"  - {f}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")

    data_vol.commit()
    print(f"Data downloaded and saved to Modal Volume 'fineweb-data'")


@app.function(
    gpu="H100:8",
    timeout=10 * 60 * 60,  # 10 hours max (baseline is much faster on 8x H100)
    volumes={TRACE_DIR: traces, DATA_DIR: data_vol},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("HF_TOKEN"),
    ],
)
def train(step: int = 1, notes: str = ""):
    """
    Run GPT-2 speedrun training for specified step.

    Uses the same Modal volumes/secrets/GPU pattern as modded-nanogpt:
    fineweb-data, ddp-traces, wandb-secret, HF_TOKEN, 8x H100.

    Steps (wall-clock much lower on 8x H100 than the original 2-GPU times):
        1: Initial baseline
        2: Architectural changes
        3: Muon optimizer
        4: Dataloading tweaks
        5: Logit soft-capping
        6: Longer sequence length

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

    # Fail fast if FineWeb data is missing on the shared volume
    print(f"Volume mounted at: {DATA_DIR}")
    print(f"Volume exists: {os.path.exists(DATA_DIR)}")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Data volume not mounted at {DATA_DIR}. "
            "Run: modal run src/runfiles/modal_runner.py::download_data"
        )
    bin_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".bin"))
    print(f"Found {len(bin_files)} .bin files on fineweb-data")
    for name in bin_files[:5]:
        print(f"  - {name}")
    if len(bin_files) > 5:
        print(f"  ... and {len(bin_files) - 5} more")
    val_file = DATA_DIR / "fineweb_val_000000.bin"
    if not val_file.exists():
        raise FileNotFoundError(
            f"Missing {val_file}. "
            "Run: modal run src/runfiles/modal_runner.py::download_data"
        )

    env = os.environ.copy()
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["TRACE_DIR"] = str(TRACE_DIR)

    run_notes = f"{step_folder}: {notes}" if notes else step_folder
    traces_before = _trace_files_in_dir(TRACE_DIR)

    print("=" * 60)
    print(f"Tyler Romero NanoGPT Speedrun - Step {step}")
    print("=" * 60)
    print(f"GPUs: 8x H100")
    print(f"Step folder: {step_folder}")
    print(f"Train file: {train_file}")
    print(f"Run script: {run_script}")
    print(f"Wandb Project: {WANDB_PROJECT}")
    print(f"Notes: {run_notes}")
    print(f"Data: {DATA_DIR}")
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
    traces_after = _trace_files_in_dir(TRACE_DIR)
    new_traces = traces_after - traces_before
    traces.commit()
    if new_traces:
        print(f"\nTraces saved to Modal Volume 'ddp-traces' ({len(new_traces)} new file(s))")
        for name in sorted(new_traces)[:5]:
            print(f"  - {name}")
        if len(new_traces) > 5:
            print(f"  ... and {len(new_traces) - 5} more")
    elif traces_after:
        print(f"\nNo new trace files this run ({len(traces_after)} existing on ddp-traces)")
    else:
        print("\nNo profiler traces on ddp-traces")
    print("Download with: modal volume get ddp-traces / ./local_traces")

    if result.returncode != 0:
        print(f"Training failed with exit code: {result.returncode}")
    else:
        print("Training completed successfully!")

    return result.returncode
