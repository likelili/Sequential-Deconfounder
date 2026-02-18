import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from sequential_deconfounder.utils.util import (
    train_dvae_with_diagnostics,
    make_fixed_user_mask,
    make_time_varying_user_mask,
)
from sequential_deconfounder.models.model import PyroDVAE


@dataclass
class TrainConfig:
    data_npz: Path
    model_out: Path
    log_out: Path
    latent_dim: int = 200
    hidden_dim: int = 256
    lr: float = 1e-3
    num_epochs: int = 200
    batch_size: int = 64
    anneal: str = "linear"
    warmup_epochs: int = 50
    cycle_length: int = 50
    cycle_ratio: float = 0.5
    seed: int = 42
    device: str = "cuda"
    mask_type: str = "none"  # none | fixed | time_varying
    holdout_frac: float = 0.1
    mask_seed: int = 0


def load_npz(path: Path) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    A = torch.tensor(data["A"], dtype=torch.float32)
    X = torch.tensor(data["X"], dtype=torch.float32)
    users = data.get("users", np.array([]))
    dates = data.get("dates", np.array([]))
    return A, X, users, dates


def train_model(cfg: TrainConfig) -> None:
    A, X, users, dates = load_npz(cfg.data_npz)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    model = PyroDVAE(
        input_dim=A.shape[-1],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        x_dim=X.shape[-1],
    )

    obs_mask = None
    if cfg.mask_type == "fixed":
        obs_mask = make_fixed_user_mask(
            U=A.shape[0],
            T=A.shape[1],
            D=A.shape[2],
            holdout_frac=cfg.holdout_frac,
            seed=cfg.mask_seed,
            device=cfg.device,
        )
    elif cfg.mask_type == "time_varying":
        obs_mask = make_time_varying_user_mask(
            U=A.shape[0],
            T=A.shape[1],
            D=A.shape[2],
            holdout_frac=cfg.holdout_frac,
            seed=cfg.mask_seed,
            device=cfg.device,
        )

    logs = train_dvae_with_diagnostics(
        model_obj=model,
        A_tensor=A,
        X_tensor=X,
        obs_mask=obs_mask,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=cfg.device,
        anneal=cfg.anneal,
        warmup_epochs=cfg.warmup_epochs,
        cycle_length=cfg.cycle_length,
        cycle_ratio=cfg.cycle_ratio,
        seed=cfg.seed,
    )

    cfg.model_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.log_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": A.shape[-1],
            "latent_dim": cfg.latent_dim,
            "hidden_dim": cfg.hidden_dim,
            "x_dim": X.shape[-1],
            "mask_type": cfg.mask_type,
            "holdout_frac": cfg.holdout_frac,
            "mask_seed": cfg.mask_seed,
        },
        cfg.model_out,
    )

    if obs_mask is not None:
        torch.save(obs_mask, cfg.model_out.with_suffix(".obs_mask.pt"))

    logs.to_csv(cfg.log_out, index=False)

    meta_path = cfg.model_out.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(cfg), handle, indent=2, default=str)
