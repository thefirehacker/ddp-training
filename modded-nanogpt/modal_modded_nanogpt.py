import modal
import os
from pathlib import Path

# Volumes - reuse existing volumes (shared with nanogpt-speedrun)
traces = modal.Volume.from_name("ddp-traces", create_if_missing=True)
data_vol = modal.Volume.from_name("fineweb-data", create_if_missing=True)
TRACE_DIR = Path("/traces")
# Mount data volume at the path where modded-nanogpt expects it
DATA_DIR = Path("/modded-nanogpt/data/fineweb10B")

WANDB_PROJECT = "modded-nanogpt-run"

# Image using the project's Dockerfile (includes nightly PyTorch)
# The Dockerfile handles CUDA 12.6, Python 3.12, and nightly torch
image = (
    modal.Image.from_dockerfile("Dockerfile")
    .add_local_dir(".", remote_path="/modded-nanogpt")
)

# Alternative image without Dockerfile (if Dockerfile causes issues)
image_alt = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "curl", "git")
    .pip_install(
        "numpy",
        "tqdm",
        "huggingface-hub",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
        "wandb",
    )
    .run_commands(
        "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126"
    )
    .add_local_dir(".", remote_path="/modded-nanogpt")
)

app = modal.App("modded-nanogpt", image=image)


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
    
    Note: This volume is shared with nanogpt-speedrun. If you already ran
    download_data from Tyler's runner, the data is already here.
    
    Args:
        num_chunks: Number of 100M token chunks to download (default 9 = 900M tokens)
                   Use 103 for full 10B tokens
    """
    import subprocess
    import sys
    import os
    
    # Debug: show volume mount
    print(f"Volume mounted at: {DATA_DIR}")
    print(f"Volume exists: {os.path.exists(DATA_DIR)}")
    if os.path.exists(DATA_DIR):
        print(f"Volume contents: {os.listdir(DATA_DIR)}")
    
    # Check if data already exists (shared from nanogpt-speedrun)
    val_file = "/modded-nanogpt/data/fineweb10B/fineweb_val_000000.bin"
    if os.path.exists(val_file):
        existing_files = len([f for f in os.listdir("/modded-nanogpt/data/fineweb10B") if f.endswith('.bin')])
        print(f"Data already exists! Found {existing_files} .bin files")
        print("Skipping download. Data was likely downloaded by nanogpt-speedrun.")
        return
    
    print(f"Downloading {num_chunks} chunks (~{num_chunks * 100}M tokens)...")
    print("This will take several minutes...")
    
    # Run with output visible
    result = subprocess.run(
        ["python", "data/cached_fineweb10B.py", str(num_chunks)],
        cwd="/modded-nanogpt",
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
    gpu="H100:8",  # Official speedrun config: 8x H100
    timeout=60 * 60,  # 1 hour (record is ~3 minutes, but compile takes ~7 min)
    volumes={TRACE_DIR: traces, DATA_DIR: data_vol},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("HF_TOKEN"),
    ],
)
def train(num_gpus: int = 8, profiler: bool = False):
    """
    Run official NanoGPT speedrun (Keller Jordan's modded-nanogpt).
    
    This runs the current world-record speedrun code on 8x H100 GPUs.
    Target: ≤3.28 validation loss on FineWeb in under 100 seconds.
    
    Args:
        num_gpus: Number of GPUs to use (default 8)
        profiler: If True, capture a short PyTorch profiler trace to /traces (ddp-traces volume)
    """
    import subprocess
    import sys
    import os

    env = os.environ.copy()
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["TRACE_DIR"] = str(TRACE_DIR)
    if profiler:
        env["ENABLE_PROFILER"] = "1"

    traces_before = _trace_files_in_dir(TRACE_DIR)

    print("=" * 60)
    print("Modded-NanoGPT Speedrun (Keller Jordan)")
    print("=" * 60)
    print(f"GPUs: {num_gpus}x H100")
    print(f"Target: ≤3.28 val loss on FineWeb")
    print(f"Wandb Project: {WANDB_PROJECT}")
    print("=" * 60)
    print("Note: torch.compile adds ~7 min latency on first run")
    print("=" * 60)

    result = subprocess.run(
        ["torchrun", "--standalone", f"--nproc_per_node={num_gpus}", "train_gpt.py"],
        cwd="/modded-nanogpt",
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
        print("Re-run with profiler enabled: modal run modal_modded_nanogpt.py::train --profiler")
    else:
        print("\nNo profiler traces on ddp-traces (enable with --profiler on train)")
    
    if result.returncode != 0:
        print(f"Training failed with exit code: {result.returncode}")
    else:
        print("Speedrun completed successfully!")
    
    return result.returncode
