#!/home/zach/miniconda3/envs/torch/bin/python
import click
import os
from torch.distributed.run import get_args_parser, run
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set pythonpath for subprocess
os.environ["PYTHONPATH"] = str(project_root)


@click.command()
@click.argument("script", type=click.Path(exists=True))
@click.option("-ng", help="Number of gpus to use")
@click.option("-ids", help="Comma separated list of gpu ids to use")
def trun(script, ng, ids):
    """Easy wrapper around torchrun"""
    args = get_args_parser().parse_args(["--nproc_per_node", ng, script])
    if ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ids
    run(args)


if __name__ == "__main__":
    trun()
