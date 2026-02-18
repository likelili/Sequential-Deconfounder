import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from sequential_deconfounder.utils.ppc_tests import run_all_ppc_tests, plot_ppc_kde_per_day
from sequential_deconfounder.utils.util import make_fixed_user_mask, make_time_varying_user_mask
from sequential_deconfounder.models.model import PyroDVAE


@dataclass
class PPCConfig:
    data_npz: Path
    model_ckpt: Path
    out_dir: Path
    holdout_steps: int = 2
    device: str = "cuda"
    mask_type: str = "time_varying"  # fixed | time_varying | none
    holdout_frac: float = 0.1
    mask_seed: int = 0
    use_train_mask: bool = True
    num_z_samples: int = 50
    batch_size: int = 128


def _load_npz(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(path, allow_pickle=True)
    A = torch.tensor(data["A"], dtype=torch.float32)
    X = torch.tensor(data["X"], dtype=torch.float32)
    return A, X


def _split_time(A: torch.Tensor, X: torch.Tensor, holdout_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if holdout_steps <= 0 or holdout_steps >= A.shape[1]:
        raise ValueError("holdout_steps must be between 1 and T-1.")
    return A[:, :-holdout_steps], X[:, :-holdout_steps], A[:, -holdout_steps:], X[:, -holdout_steps:]


def _load_model(ckpt_path: Path, device: str) -> PyroDVAE:
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


def run_ppc(cfg: PPCConfig) -> None:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    A, X = _load_npz(cfg.data_npz)

    model = _load_model(cfg.model_ckpt, cfg.device)

    @contextmanager
    def _cwd(path: Path):
        prev = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(prev)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = cfg.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    obs_mask = None
    mask_path = cfg.model_ckpt.with_suffix(".obs_mask.pt")
    if cfg.use_train_mask and mask_path.exists():
        obs_mask = torch.load(mask_path, map_location=cfg.device)
    elif cfg.mask_type == "fixed":
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

    if obs_mask is None:
        raise ValueError("obs_mask is required for masked PPC but was not created.")

    with _cwd(figures_dir):
        results = run_all_ppc_tests(
            model_obj=model,
            A=A,
            X=X,
            obs_mask=obs_mask,
            num_z_samples=cfg.num_z_samples,
            batch_size=cfg.batch_size,
        )
        p_values = results["test1_temporal_masked_ppc"]["ppc_per_day"]
        plot_ppc_kde_per_day(p_values)

    out_path = cfg.out_dir / "ppc_results.json"
    config_dict = asdict(cfg)
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in config_dict.items()}
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": config_dict, "results": str(results)},
            handle,
            indent=2,
        )
