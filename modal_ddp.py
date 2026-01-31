import modal
from pathlib import Path

# Create a persistent volume for traces
traces = modal.Volume.from_name("ddp-traces", create_if_missing=True)
TRACE_DIR = Path("/traces")

# Define container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch~=2.5.1",
        "accelerate",
        "transformers",
        "datasets",
        "numpy",
        "wandb",
    )
    .add_local_dir("training_utils", remote_path="/root/training_utils")
    .add_local_file("ddp.py", remote_path="/root/ddp.py")
)

app = modal.App("ddp-training", image=image)


@app.function(
    gpu="L40S:2",
    timeout=60 * 60,
    volumes={TRACE_DIR: traces},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_single_node():
    """Single-node multi-GPU training with 2x L40S GPUs"""
    import subprocess
    import sys
    import os

    # Pass trace directory to ddp.py via environment variable
    env = os.environ.copy()
    env["TRACE_DIR"] = str(TRACE_DIR)

    subprocess.run(
        ["torchrun", "--nproc_per_node=2", "ddp.py"],
        cwd="/root",
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        check=True,
    )

    # Commit volume to persist traces
    traces.commit()
    print(f"Traces saved to Modal Volume 'ddp-traces'")


# Optional: For multi-node training (requires beta access)
# n_nodes = 2
#
# @app.function(gpu="H100:8", timeout=60 * 60 * 24)
# @modal.experimental.clustered(size=n_nodes, rdma=True)
# def train_multi_node():
#     from torch.distributed.run import parse_args, run
#     cluster_info = modal.experimental.get_cluster_info()
#     run(parse_args([
#         f"--nnodes={n_nodes}",
#         f"--node-rank={cluster_info.rank}",
#         f"--master-addr={cluster_info.container_ips[0]}",
#         "--nproc-per-node=8",
#         "--master-port=1234",
#         "ddp.py",
#     ]))
