import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full Sequential Deconfounder pipeline.")
    parser.add_argument("--preprocess_config", default="configs/base.yaml")
    parser.add_argument("--train_config", default="configs/train.yaml")
    parser.add_argument("--ppc_config", default="configs/ppc.yaml")
    parser.add_argument("--effects_config", default="configs/effects.yaml")
    return parser


def run_cmd(cmd: str, env: dict) -> None:
    print(f"\n[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True, env=env)


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}{os.pathsep}{env.get('PYTHONPATH','')}"

    run_cmd(f"{sys.executable} scripts/preprocess.py --config {args.preprocess_config}", env)
    run_cmd(f"{sys.executable} scripts/train.py --config {args.train_config}", env)
    run_cmd(f"{sys.executable} scripts/run_ppc.py --config {args.ppc_config}", env)
    run_cmd(f"{sys.executable} scripts/estimate_effects.py --config {args.effects_config}", env)


if __name__ == "__main__":
    main()
