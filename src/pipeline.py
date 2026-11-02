"""CLI entry point. Run from the project root: python -m src.pipeline --stage all."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .common import EXTRACTORS, ROOT, dirs, read_config, save_json, runtime_metadata, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["all", "prepare", "extract", "benchmark",
                        "warmup", "freeze", "freeze-finalize", "plots", "audit", "report"])
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--model", choices=list(EXTRACTORS))
    parser.add_argument("--policy", type=int, help="one of config['unfreeze_last']")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.policy is not None and args.policy not in config["unfreeze_last"]:
        parser.error(f"--policy must be one of {config['unfreeze_last']}")
    if args.stage == "all":
        stages = [("prepare", [])] + [("extract", ["--model", name]) for name in EXTRACTORS]
        stages += [("benchmark", []), ("warmup", [])]
        stages += [("freeze", ["--policy", str(k)]) for k in config["unfreeze_last"]]
        stages += [("freeze-finalize", []), ("plots", []), ("audit", []), ("report", [])]
        env = dict(os.environ, PYTHONHASHSEED=str(config["seed"]), PYTHONUNBUFFERED="1")
        for name, extra in stages:
            # Separate processes release TensorFlow's allocator between policies.
            subprocess.run([sys.executable, "-m", "src.pipeline", "--stage", name,
                            "--config", str(Path(args.config).resolve()), *extra],
                           cwd=ROOT, env=env, check=True)
        return
    # Rendering saved results needs no TensorFlow, pretrained weights or JPEG download.
    if args.stage in ("plots", "report"):
        dirs()
    else:
        setup(config)
    started = perf_counter()
    if args.stage == "prepare":
        from .data import prepare
        save_json(ROOT / "results/runtime.json", runtime_metadata(config))
        prepare(config)
    elif args.stage == "extract":
        from .features import extract
        if not args.model:
            parser.error("--model is required for extract")
        extract(config, args.model)
    elif args.stage == "benchmark":
        from .benchmark import run
        run(config)
    elif args.stage == "warmup":
        from .freeze import warmup
        warmup(config)
    elif args.stage == "freeze":
        from .freeze import run_policy
        if args.policy is None:
            parser.error("--policy is required for freeze")
        run_policy(config, args.policy)
    elif args.stage == "freeze-finalize":
        from .freeze import finalize
        finalize(config)
    elif args.stage == "plots":
        from .visuals import generate
        generate(config)
    elif args.stage == "audit":
        from .audit import audit
        audit(config)
    elif args.stage == "report":
        from .reporting import generate
        generate(config)
    print(f"Completed {args.stage}: {perf_counter()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
