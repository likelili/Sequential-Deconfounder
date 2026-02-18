from typing import Tuple

import torch
import pyro.poutine as poutine

from sequential_deconfounder.utils.outcome import estimate_temporal_population_effects


def infer_latent_trajectories(model, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Infer Z_{1:T} with the guide and stack into (U, T, latent_dim).
    """
    with torch.no_grad():
        guide_trace = poutine.trace(model.guide).get_trace(A, X)
        z_sites = sorted(
            [k for k in guide_trace.nodes.keys() if k.startswith("z_")],
            key=lambda s: int(s.split("_")[1]),
        )
        z_list = [guide_trace.nodes[name]["value"] for name in z_sites]
        Z = torch.stack(z_list, dim=1)
    return Z


def estimate_population_effects(
    A: torch.Tensor,
    Z: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    decoder_model,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return estimate_temporal_population_effects(A, Z, X, Y, decoder_model)


__all__ = ["infer_latent_trajectories", "estimate_population_effects"]
