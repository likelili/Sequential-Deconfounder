import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sequential_deconfounder.models.model import PyroDVAE
from sequential_deconfounder.outcomes.effects import infer_latent_trajectories, estimate_population_effects
from sequential_deconfounder.outcomes.qte import estimate_dual_causal_effects
from sequential_deconfounder.outcomes.residualize import estimate_outcome_model_residualized
from sequential_deconfounder.utils.outcome import (
    build_purchase_tables,
    build_Y_tensors_from_buy_day,
    estimate_temporal_population_uncorrected,
    plot_beta_distributions,
    analyze_top_categories,
    compare_deconfounding,
)


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    A = torch.tensor(data["A"], dtype=torch.float32)
    X = torch.tensor(data["X"], dtype=torch.float32)
    users = data.get("users", np.array([]))
    dates = data.get("dates", np.array([]))
    return A, X, users.tolist(), dates.tolist()


def load_model(ckpt_path: Path, device: str) -> PyroDVAE:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PyroDVAE(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        hidden_dim=ckpt["hidden_dim"],
        x_dim=ckpt["x_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


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
    parser = argparse.ArgumentParser(description="Estimate causal effects from trained model.")
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    parser.add_argument("--data_npz", help="Path to dvae_inputs.npz")
    parser.add_argument("--model_ckpt", help="Path to trained model checkpoint")
    parser.add_argument("--buy_csv", help="Path to buy.csv (from preprocessing)")
    parser.add_argument("--out_dir", help="Directory to save effects")
    parser.add_argument("--fig_dir", help="Directory to save figures")
    parser.add_argument("--quantile", type=float, default=0.95, help="Quantile for QTE")
    parser.add_argument(
        "--quantiles",
        nargs="*",
        type=float,
        default=None,
        help="List of quantiles for QTE curves (dual only)",
    )
    parser.add_argument(
        "--qte_method",
        choices=["dual", "residualized"],
        default="dual",
        help="QTE estimator: dual (fast) or residualized (notebook)",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = _apply_config(build_parser().parse_args())
    required = ["data_npz", "model_ckpt", "buy_csv", "out_dir", "fig_dir"]
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    A, X, users, dates = load_npz(Path(args.data_npz))
    A = A.to(device)
    X = X.to(device)
    model = load_model(Path(args.model_ckpt), device)

    Z = infer_latent_trajectories(model, A, X)

    buy = pd.read_csv(args.buy_csv)
    if "date" not in buy.columns and "time_stamp" in buy.columns:
        buy = buy.rename(columns={"time_stamp": "date"})

    buy_day = build_purchase_tables(buy, binary=True, day_col="date")
    _, Y_next, Y_cum = build_Y_tensors_from_buy_day(buy_day, users, dates)
    Y_next_t = torch.tensor(Y_next, dtype=torch.float32, device=device)
    Y_cum_t = torch.tensor(Y_cum, dtype=torch.float32, device=device)

    def _align_time(A_t, Z_t, X_t, Y_t):
        t_len = Y_t.shape[1]
        return A_t[:, :t_len], Z_t[:, :t_len], X_t[:, :t_len], Y_t

    A_next, Z_next, X_next, Y_next_t = _align_time(A, Z, X, Y_next_t)
    A_cum, Z_cum, X_cum, Y_cum_t = _align_time(A, Z, X, Y_cum_t)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    beta_inst_pop_next, beta_cum_pop_next = estimate_population_effects(
        A=A_next,
        Z=Z_next,
        X=X_next,
        Y=Y_next_t,
        decoder_model=model.emission,
    )
    beta_inst_pop_cum, beta_cum_pop_cum = estimate_population_effects(
        A=A_cum,
        Z=Z_cum,
        X=X_cum,
        Y=Y_cum_t,
        decoder_model=model.emission,
    )

    if args.qte_method == "dual":
        quantiles = args.quantiles or [0.05, 0.25, 0.5, 0.75, 0.95]
        betas_inst = {}
        betas_cum = {}
        for q in quantiles:
            beta_inst_q, beta_cum_q = estimate_dual_causal_effects(
                A=A_next,
                Z=Z_next,
                X=X_next,
                Y=Y_next_t,
                decoder_model=model.emission,
                quantile=q,
            )
            betas_inst[q] = beta_inst_q
            betas_cum[q] = beta_cum_q
            np.save(out_dir / f"beta_inst_qte_next_q{q}.npy", beta_inst_q)
            np.save(out_dir / f"beta_cum_qte_next_q{q}.npy", beta_cum_q)
    else:
        quantiles = [args.quantile]
        betas_inst = {}
        betas_cum = {}
        beta_qte_next = estimate_outcome_model_residualized(
            A=A_next,
            Z=Z_next,
            X=X_next,
            Y=Y_next_t,
            decoder_model=model.emission,
            quantile=args.quantile,
        )
        beta_qte_cum = estimate_outcome_model_residualized(
            A=A_cum,
            Z=Z_cum,
            X=X_cum,
            Y=Y_cum_t,
            decoder_model=model.emission,
            quantile=args.quantile,
        )
        betas_inst[args.quantile] = beta_qte_next
        betas_cum[args.quantile] = beta_qte_cum
        np.save(out_dir / f"beta_qte_resid_next_q{args.quantile}.npy", beta_qte_next)
        np.save(out_dir / f"beta_qte_resid_cum_q{args.quantile}.npy", beta_qte_cum)

    np.save(out_dir / "beta_inst_pop_next.npy", beta_inst_pop_next)
    np.save(out_dir / "beta_cum_pop_next.npy", beta_cum_pop_next)
    np.save(out_dir / "beta_inst_pop_cum.npy", beta_inst_pop_cum)
    np.save(out_dir / "beta_cum_pop_cum.npy", beta_cum_pop_cum)
    import matplotlib.pyplot as plt

    def _save_hist(values, title, filename):
        plt.figure()
        plt.hist(values, bins=50, alpha=0.8)
        plt.title(title)
        plt.xlabel("Effect size")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=300)
        plt.close()

    _save_hist(beta_inst_pop_next, "Instantaneous effects (next-day, population)", "beta_inst_pop_next.png")
    _save_hist(beta_cum_pop_next, "Cumulative effects (next-day, population)", "beta_cum_pop_next.png")
    _save_hist(beta_inst_pop_cum, "Instantaneous effects (cumulative outcome, population)", "beta_inst_pop_cum.png")
    _save_hist(beta_cum_pop_cum, "Cumulative effects (cumulative outcome, population)", "beta_cum_pop_cum.png")
    plot_beta_distributions(betas_inst)
    plot_beta_distributions(betas_cum)

    A_mean = A_next.mean(dim=(0, 1))
    top_cum_df = analyze_top_categories({"Cumulative": beta_cum_pop_next}, A_mean=A_mean)
    top_cum_df.to_csv(out_dir / "top_cumulative_categories.csv", index=False)

    beta_inst_unconf, beta_cum_unconf = estimate_temporal_population_uncorrected(
        A=A_next,
        X=X_next,
        Y=Y_next_t,
    )
    compare_deconfounding(beta_inst_pop_next, beta_inst_unconf)
    compare_deconfounding(beta_cum_pop_next, beta_cum_unconf)

    for q, b in betas_inst.items():
        _save_hist(b, f"QTE inst (next-day, q={q})", f"beta_inst_qte_next_q{q}.png")
    for q, b in betas_cum.items():
        _save_hist(b, f"QTE cum (next-day, q={q})", f"beta_cum_qte_next_q{q}.png")


if __name__ == "__main__":
    main()
