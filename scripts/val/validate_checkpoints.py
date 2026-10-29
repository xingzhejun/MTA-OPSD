#!/usr/bin/env python3
"""Validate one or more verl FSDP checkpoints without entering training."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml


STEP_RE = re.compile(r"^global_step_(\d+)$")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean, got {value!r}")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def discover_steps(checkpoint_dir: Path) -> List[int]:
    steps = []
    for child in checkpoint_dir.iterdir():
        match = STEP_RE.match(child.name) if child.is_dir() else None
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def parse_steps(spec: str, available: List[int]) -> List[int]:
    if spec.strip().lower() == "all":
        return available
    requested = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested checkpoint steps do not exist: {missing}")
    return requested


def checkpoint_world_size(actor_dir: Path) -> int:
    fsdp_path = actor_dir / "fsdp_config.json"
    if not fsdp_path.is_file():
        raise FileNotFoundError(f"Missing FSDP metadata: {fsdp_path}")
    with fsdp_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    world_size = int(metadata["world_size"])
    for rank in range(world_size):
        shard = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not shard.is_file():
            raise FileNotFoundError(f"Missing model shard: {shard}")
    return world_size


def merger_environment(repo_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(repo_root / "third_party" / "verl"), str(repo_root)]
    user = env.get("USER", "")
    amd_site = Path(f"/tmp/{user}/python-envs/dev/lib/python3.12/site-packages")
    opt_site = Path("/opt/venv/lib/python3.12/site-packages")
    python_paths.extend(str(path) for path in (amd_site, opt_site) if path.exists())
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def merge_checkpoint(repo_root: Path, actor_dir: Path, target_dir: Path, dry_run: bool) -> None:
    marker = target_dir / "config.json"
    if marker.is_file():
        print(f"[merge] reuse existing model: {target_dir}", flush=True)
        return
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(target_dir),
    ]
    print("[merge] " + " ".join(command), flush=True)
    if not dry_run:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=repo_root, env=merger_environment(repo_root), check=True)


def newest_run_dir(step_output_dir: Path, before: Set[Path]) -> Path:
    candidates = {path for path in step_output_dir.iterdir() if path.is_dir()} - before
    if not candidates:
        candidates = {path for path in step_output_dir.iterdir() if path.is_dir()}
    if not candidates:
        raise RuntimeError(f"Validation launcher produced no run directory under {step_output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def extract_metrics(metrics_path: Path) -> Dict[str, float]:
    selected = {}  # type: Dict[str, float]
    if not metrics_path.is_file():
        return selected
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = payload.get("data", payload)
            current = {key: value for key, value in data.items() if key.startswith("val-core/")}
            if current:
                selected = current
    return selected


def write_result(step: int, run_dir: Path, result_path: Path) -> Dict[str, Any]:
    result = {
        "step": step,
        "run_dir": str(run_dir),
        "metrics": extract_metrics(run_dir / "metrics.jsonl"),
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def write_summary(validation_root: Path, results: List[Dict[str, Any]]) -> None:
    metric_keys = sorted({key for result in results for key in result.get("metrics", {})})
    summary_path = validation_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "run_dir", *metric_keys])
        writer.writeheader()
        for result in sorted(results, key=lambda item: int(item["step"])):
            row = {"step": result["step"], "run_dir": result["run_dir"]}
            row.update(result.get("metrics", {}))
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", help="Comma-separated global steps, or 'all'.")
    parser.add_argument("--checkpoint-dir", type=Path)
    merge_group = parser.add_mutually_exclusive_group()
    merge_group.add_argument("--merge", dest="merge", action="store_true")
    merge_group.add_argument("--no-merge", dest="merge", action="store_false")
    parser.set_defaults(merge=None)
    parser.add_argument("--force", action="store_true", help="Re-run completed steps.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only.")
    args, hydra_overrides = parser.parse_known_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_dir = (args.checkpoint_dir or Path(config["checkpoint_dir"])).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    available = discover_steps(checkpoint_dir)
    if not available:
        raise RuntimeError(f"No global_step_* checkpoints found under {checkpoint_dir}")
    step_spec = args.steps or str(config.get("checkpoint_steps", "all"))
    steps = parse_steps(step_spec, available)
    use_merge = parse_bool(config.get("merge_checkpoint", False)) if args.merge is None else args.merge
    skip_completed = parse_bool(config.get("skip_completed", True)) and not args.force
    validation_root = checkpoint_dir / str(config.get("validation_output_dir", "validation"))
    if not args.dry_run:
        validation_root.mkdir(parents=True, exist_ok=True)

    print(f"checkpoints: {checkpoint_dir}", flush=True)
    print(f"steps: {steps}", flush=True)
    print(f"merge_checkpoint: {use_merge}", flush=True)
    results = []  # type: List[Dict[str, Any]]

    for step in steps:
        checkpoint = checkpoint_dir / f"global_step_{step}"
        actor_dir = checkpoint / "actor"
        saved_world_size = checkpoint_world_size(actor_dir)
        step_output_dir = validation_root / f"global_step_{step}"
        if not args.dry_run:
            step_output_dir.mkdir(parents=True, exist_ok=True)
        result_path = step_output_dir / "validation_result.json"
        if skip_completed and result_path.is_file():
            print(f"[step {step}] already complete; skipping", flush=True)
            with result_path.open("r", encoding="utf-8") as handle:
                results.append(json.load(handle))
            continue

        derived = dict(config)
        derived["output_dir"] = str(step_output_dir)
        derived["experiment_name"] = f"checkpoint-validation-step-{step}"
        derived["val_before_train"] = True
        derived["use_swanlab"] = False

        # The shared launcher currently derives train_batch_size as
        # per_device_batch_size * (4 / tensor_parallel_size).  verl constructs
        # that dataloader before checking val_only and asserts it is non-empty.
        tp_size = int(derived.get("vllm_tensor_parallel_size", 1))
        if tp_size <= 0 or 4 % tp_size != 0:
            raise ValueError(f"vllm_tensor_parallel_size must divide 4, got {tp_size}")
        init_train_batch = int(derived.get("per_device_batch_size", 1)) * (4 // tp_size)
        derived["max_train_samples"] = max(int(derived.get("max_train_samples", 0)), init_train_batch)

        if use_merge:
            merged_root = checkpoint_dir / str(config.get("merged_model_dir", "merged_hf"))
            model_path = merged_root / f"global_step_{step}"
            merge_checkpoint(repo_root, actor_dir, model_path, args.dry_run)
            derived["model_path"] = str(model_path)
            derived["n_gpus_per_node"] = int(config.get("merge_eval_gpus", 1))
            resume_overrides = ["trainer.resume_mode=disable"]
        else:
            configured_gpus = int(config.get("n_gpus_per_node", saved_world_size))
            if configured_gpus != saved_world_size:
                raise ValueError(
                    f"global_step_{step} was saved with world_size={saved_world_size}, but "
                    f"n_gpus_per_node={configured_gpus}. Direct loading requires them to match."
                )
            derived["n_gpus_per_node"] = saved_world_size
            resume_overrides = [
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path={checkpoint}",
                "actor_rollout_ref.actor.checkpoint.load_contents=[model]",
            ]

        # Legacy workers validate training batch relationships during startup,
        # even though val_only returns before the first optimizer step.  The
        # shared launcher fixes ppo_micro_batch_size_per_gpu at 4 and treats
        # ppo_mini_batch_size as a global value, so it must cover every DP rank.
        eval_gpus = int(derived["n_gpus_per_node"])
        actor_dp_size = eval_gpus // tp_size
        minimum_mini_batch = 4 * actor_dp_size
        configured_mini_batch = int(derived.get("ppo_mini_batch_size", minimum_mini_batch))
        if configured_mini_batch < minimum_mini_batch or configured_mini_batch % minimum_mini_batch != 0:
            print(
                f"[step {step}] adjusting ppo_mini_batch_size "
                f"from {configured_mini_batch} to {minimum_mini_batch} for val-only initialization",
                flush=True,
            )
            derived["ppo_mini_batch_size"] = minimum_mini_batch

        # Orchestration-only keys need not be exposed to the shared launcher.
        for key in (
            "checkpoint_dir",
            "checkpoint_steps",
            "validation_output_dir",
            "skip_completed",
            "merge_checkpoint",
            "merge_eval_gpus",
            "merged_model_dir",
        ):
            derived.pop(key, None)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"val_step_{step}_", encoding="utf-8", delete=False
        ) as handle:
            yaml.safe_dump(derived, handle, sort_keys=False)
            derived_path = Path(handle.name)

        command = [
            "bash",
            str(repo_root / "scripts" / "_run_verl.sh"),
            str(derived_path),
            "trainer.val_before_train=true",
            "trainer.val_only=true",
            "trainer.logger=[console,file]",
            *resume_overrides,
            *hydra_overrides,
        ]
        before = {path for path in step_output_dir.iterdir() if path.is_dir()} if step_output_dir.exists() else set()
        print(f"[step {step}] " + " ".join(command), flush=True)
        try:
            if args.dry_run:
                continue
            subprocess.run(command, cwd=repo_root, check=True)
            run_dir = newest_run_dir(step_output_dir, before)
            result = write_result(step, run_dir, result_path)
            if not result["metrics"]:
                raise RuntimeError(f"No val-core metrics found in {run_dir / 'metrics.jsonl'}")
            results.append(result)
        finally:
            if derived_path.exists():
                derived_path.unlink()

    if not args.dry_run:
        # Include completed checkpoints not selected in this invocation as well.
        known = {int(result["step"]): result for result in results}
        for result_path in validation_root.glob("global_step_*/validation_result.json"):
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            known[int(result["step"])] = result
        write_summary(validation_root, list(known.values()))
        print(f"summary: {validation_root / 'summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())