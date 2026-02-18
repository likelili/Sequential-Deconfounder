import argparse
from pathlib import Path

from sequential_deconfounder.inference.train import TrainConfig, train_model


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
    parser = argparse.ArgumentParser(description="Train Sequential Deconfounder model.")
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    parser.add_argument("--data_npz", help="Path to dvae_inputs.npz")
    parser.add_argument("--model_out", help="Path to save model checkpoint")
    parser.add_argument("--log_out", help="Path to save training log CSV")
    parser.add_argument("--latent_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--anneal", choices=["linear", "cyclical", "none"], default="linear")
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--cycle_length", type=int, default=50)
    parser.add_argument("--cycle_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask_type", choices=["none", "fixed", "time_varying"], default="none")
    parser.add_argument("--holdout_frac", type=float, default=0.1)
    parser.add_argument("--mask_seed", type=int, default=0)
    return parser


def main() -> None:
    args = _apply_config(build_parser().parse_args())
    required = ["data_npz", "model_out", "log_out"]
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    cfg = TrainConfig(
        data_npz=Path(args.data_npz),
        model_out=Path(args.model_out),
        log_out=Path(args.log_out),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        anneal=args.anneal,
        warmup_epochs=args.warmup_epochs,
        cycle_length=args.cycle_length,
        cycle_ratio=args.cycle_ratio,
        seed=args.seed,
        device=args.device,
        mask_type=args.mask_type,
        holdout_frac=args.holdout_frac,
        mask_seed=args.mask_seed,
    )
    train_model(cfg)


if __name__ == "__main__":
    main()
