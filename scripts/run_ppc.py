import argparse
from pathlib import Path

from sequential_deconfounder.diagnostics.ppc import PPCConfig, run_ppc


def _load_yaml_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for --config. Install with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg = _load_yaml_config(Path(args.config))
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PPC diagnostics.")
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    parser.add_argument("--data_npz", help="Path to dvae_inputs.npz")
    parser.add_argument("--model_ckpt", help="Path to trained model checkpoint")
    parser.add_argument("--out_dir", help="Directory to save PPC results")
    parser.add_argument("--holdout_steps", type=int, default=2, help="Held-out time steps")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask_type", choices=["none", "fixed", "time_varying"], default="fixed")
    parser.add_argument("--holdout_frac", type=float, default=0.1)
    parser.add_argument("--mask_seed", type=int, default=0)
    parser.add_argument("--use_train_mask", action="store_true")
    parser.add_argument("--num_z_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser


def main() -> None:
    args = _apply_config(build_parser().parse_args())
    required = ["data_npz", "model_ckpt", "out_dir"]
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    cfg = PPCConfig(
        data_npz=Path(args.data_npz),
        model_ckpt=Path(args.model_ckpt),
        out_dir=Path(args.out_dir),
        holdout_steps=args.holdout_steps,
        device=args.device,
        mask_type=args.mask_type,
        holdout_frac=args.holdout_frac,
        mask_seed=args.mask_seed,
        use_train_mask=args.use_train_mask,
        num_z_samples=args.num_z_samples,
        batch_size=args.batch_size,
    )
    run_ppc(cfg)


if __name__ == "__main__":
    main()
