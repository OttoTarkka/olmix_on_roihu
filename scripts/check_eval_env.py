#!/usr/bin/env python3
"""Check that the olmix eval venv works: packages, GPU, olmix imports, BPB tasks.

Run inside the venv, e.g. at the end of hpc/roihu/prepare_eval_venv.sh or
later as a smoke test:

    "$OLMIX_PYTHON" scripts/check_eval_env.py --task-dir eval_tasks/bpb --require-gpu

Every check raises on failure, so a zero exit status means all passed. Nothing
is downloaded: the task check only resolves task names, not datasets.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import logging
from pathlib import Path

import yaml

LOGGER = logging.getLogger("check_eval_env")
DEFAULT_GROUPS = ("olmix_bpb_deu", "olmix_bpb_spa", "olmix_bpb_hun")
# The olmix entry points the scripts in this repository use.
OLMIX_MODULES = ("olmix.aliases", "olmix.generate.synthesize_mixture", "olmix.cli", "olmix.fit.cli")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True, help="lm-eval include path with the BPB tasks.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Task groups that must resolve, each defined in <task-dir>/<group>.yaml.",
    )
    parser.add_argument("--require-gpu", action="store_true", help="Fail if PyTorch sees no CUDA device.")
    return parser.parse_args()


def check_versions() -> None:
    """Log the key package versions and enforce the ones the tasks depend on.

    Raises:
        RuntimeError: if datasets is 4 or newer (FLORES needs loading scripts).
    """
    for package in ("torch", "numpy", "transformers", "datasets", "lm_eval", "olmix", "ai2-olmo-core"):
        LOGGER.info("%s %s", package, importlib.metadata.version(package))
    datasets_major = int(importlib.metadata.version("datasets").split(".")[0])
    if datasets_major >= 4:
        raise RuntimeError("datasets >= 4 cannot load facebook/flores; the venv must have datasets < 4")


def check_gpu(require_gpu: bool) -> None:
    """Log the visible GPU, or fail if one is required and none is visible.

    Args:
        require_gpu: whether a missing CUDA device is an error.

    Raises:
        RuntimeError: if ``require_gpu`` is set and no CUDA device is visible.
    """
    import torch

    if torch.cuda.is_available():
        LOGGER.info("GPU: %s (CUDA %s)", torch.cuda.get_device_name(0), torch.version.cuda)
    elif require_gpu:
        raise RuntimeError("PyTorch sees no CUDA device")
    else:
        LOGGER.info("No GPU visible (not required)")


def check_olmix() -> None:
    """Import every olmix module the repository's scripts use."""
    for module in OLMIX_MODULES:
        importlib.import_module(module)
        LOGGER.info("imported %s", module)
    from olmix.generate.synthesize_mixture import generate_weights_dirichlet

    LOGGER.info("found %s", generate_weights_dirichlet.__name__)


def check_tasks(task_dir: Path, groups: list[str]) -> None:
    """Check that lm-eval finds each group and every task the group lists.

    Args:
        task_dir: the include path holding the BPB task YAMLs.
        groups: group names; each must have ``<task_dir>/<group>.yaml``.

    Raises:
        FileNotFoundError: if the task directory or a group file is missing.
        RuntimeError: if lm-eval does not know a group or one of its tasks.
    """
    from lm_eval.tasks import TaskManager

    if not task_dir.is_dir():
        raise FileNotFoundError(f"no task directory: {task_dir}")
    manager = TaskManager(include_path=str(task_dir.resolve()))
    known = set(manager.all_tasks)
    for group in groups:
        group_file = task_dir / f"{group}.yaml"
        with group_file.open(encoding="utf-8") as handle:
            tasks = yaml.safe_load(handle)["task"]
        missing = [name for name in [group, *tasks] if name not in known]
        if missing:
            raise RuntimeError(f"lm-eval does not know {missing} (group {group})")
        LOGGER.info("%s: %d tasks resolve", group, len(tasks))


def main() -> None:
    """Run every check; any failure raises."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    check_versions()
    check_gpu(args.require_gpu)
    check_olmix()
    check_tasks(args.task_dir, args.groups)
    LOGGER.info("All checks passed")


if __name__ == "__main__":
    main()
