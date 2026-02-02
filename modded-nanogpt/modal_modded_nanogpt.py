import modal
from pathlib import Path

# Volumes - reuse existing volumes
traces = modal.Volume.from_name("ddp-traces", create_if_missing=True)
data_vol = modal.Volume.from_name("fineweb-data", create_if_missing=True)
TRACE_DIR = Path("/traces")
DATA_DIR = Path("/data")

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


@app.function(
    gpu="H100:8",
    timeout=2 * 60 * 60,
    volumes={DATA_DIR: data_vol},
)
def download_data(num_chunks: int = 9):
    """
    Download FineWeb10B data to persistent volume.
    
    Args:
        num_chunks: Number of 100M token chunks to download (default 9 = 900M tokens)
                   Use 103 for full 10B tokens
    """
    import subprocess
    
    print(f"Downloading {num_chunks} chunks (~{num_chunks * 100}M tokens)...")
    
    subprocess.run(
        ["python", "data/cached_fineweb10B.py", str(num_chunks)],
        cwd="/modded-nanogpt",
        check=True,
    )
    data_vol.commit()
    print(f"Data downloaded and saved to Modal Volume 'fineweb-data'")


@app.function(
    gpu="H100:8",  # Official speedrun config: 8x H100
    timeout=60 * 60,  # 1 hour (record is ~3 minutes, but compile takes ~7 min)
    volumes={TRACE_DIR: traces, DATA_DIR: data_vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(num_gpus: int = 8):
    """
    Run official NanoGPT speedrun (Keller Jordan's modded-nanogpt).
    
    This runs the current world-record speedrun code on 8x H100 GPUs.
    Target: ≤3.28 validation loss on FineWeb in under 100 seconds.
    
    Args:
        num_gpus: Number of GPUs to use (default 8)
    """
    import subprocess
    import sys
    import os

    env = os.environ.copy()
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["TRACE_DIR"] = str(TRACE_DIR)

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
    traces.commit()
    print(f"\nTraces saved to Modal Volume 'ddp-traces'")
    
    if result.returncode != 0:
        print(f"Training failed with exit code: {result.returncode}")
    else:
        print("Speedrun completed successfully!")
    
    return result.returncode
