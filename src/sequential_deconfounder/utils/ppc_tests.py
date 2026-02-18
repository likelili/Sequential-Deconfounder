"""
Posterior Predictive Checks (PPC) for the Sequential Deconfounder.

Implements three PPC tests:
1) Temporal masked PPC per day
2) Conditional independence of causes given Z
3) Static vs temporal comparison
"""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from pyro.infer import Predictive
import pyro.distributions as dist


def _compute_rate(model_obj, Z, X):
    ZX = torch.cat([Z, X], dim=-1)
    rate = F.softplus(model_obj.emission(ZX))
    return torch.clamp(rate, min=1e-6, max=1e3)


@torch.no_grad()
def infer_posterior_z(model_obj, A, X, obs_mask=None) -> torch.Tensor:
    """
    Infer a single posterior sample Z_{1:T} from the guide.
    Returns: (U, T, latent_dim)
    """
    import pyro.poutine as poutine

    trace = poutine.trace(model_obj.guide).get_trace(A, X, obs_mask=obs_mask)
    z_names = sorted([k for k in trace.nodes if k.startswith("z_")], key=lambda s: int(s.split("_")[1]))
    z_list = [trace.nodes[name]["value"] for name in z_names]
    return torch.stack(z_list, dim=1)


@torch.no_grad()
def ppc_discrepancy_per_user(
    model_obj,
    A,
    X,
    obs_mask,
    num_z_samples=50,
    batch_size=128,
):
    """
    Compute PPC discrepancies with masked causes.
    Returns:
      t_obs: (U, T) mean log-prob for held-out entries
      t_rep: (U, T, S) replicated log-prob for held-out entries
    """
    model_obj.eval()
    device = next(model_obj.parameters()).device
    A = A.to(device)
    X = X.to(device)
    obs_mask = obs_mask.to(device)

    U = A.shape[0]
    t_obs_list = []
    t_rep_list = []

    for i in range(0, U, batch_size):
        A_b = A[i : i + batch_size]
        X_b = X[i : i + batch_size]
        M_b = obs_mask[i : i + batch_size]

        predictive = Predictive(
            model_obj.guide,
            num_samples=num_z_samples,
            return_sites=None,
        )
        guide_samples = predictive(A_b, X_b, obs_mask=M_b)

        z_names = sorted(
            [k for k in guide_samples.keys() if k.startswith("z_")],
            key=lambda s: int(s.split("_")[1]),
        )
        Z = torch.stack([guide_samples[name] for name in z_names], dim=2)  # (S, B, T, latent)

        S = Z.shape[0]
        X_expand = X_b.unsqueeze(0).expand(S, -1, -1, -1)
        rate = _compute_rate(model_obj, Z, X_expand)

        held_mask = (~M_b).to(rate.device)
        held_expand = held_mask.unsqueeze(0).expand(S, -1, -1, -1)

        A_expand = A_b.unsqueeze(0).expand(S, -1, -1, -1)
        logp_obs = dist.Poisson(rate).log_prob(A_expand) * held_expand
        t_obs = logp_obs.sum(dim=3).mean(dim=0)  # (B, T)

        A_rep = dist.Poisson(rate).sample()
        logp_rep = dist.Poisson(rate).log_prob(A_rep) * held_expand
        t_rep = logp_rep.sum(dim=3).permute(1, 2, 0)  # (B, T, S)

        t_obs_list.append(t_obs.cpu())
        t_rep_list.append(t_rep.cpu())

    t_obs_all = torch.cat(t_obs_list, dim=0)
    t_rep_all = torch.cat(t_rep_list, dim=0)
    return t_obs_all, t_rep_all


def predictive_score(t_obs, t_rep) -> float:
    """
    Posterior predictive p-value averaged over users/time.
    """
    if isinstance(t_obs, np.ndarray):
        t_obs = torch.tensor(t_obs)
    if isinstance(t_rep, np.ndarray):
        t_rep = torch.tensor(t_rep)

    t_obs_exp = t_obs.unsqueeze(-1)
    pvals = (t_rep <= t_obs_exp).float().mean(dim=-1)
    return float(pvals.mean().item())


def compute_predictive_scores_per_time(t_obs, t_rep):
    """
    Compute per-time PPC scores.
    """
    if isinstance(t_obs, np.ndarray):
        t_obs = torch.tensor(t_obs)
    if isinstance(t_rep, np.ndarray):
        t_rep = torch.tensor(t_rep)

    t_obs_exp = t_obs.unsqueeze(-1)
    pvals = (t_rep <= t_obs_exp).float().mean(dim=-1)  # (U, T)
    return pvals.mean(dim=0).tolist()


def plot_ppc_kde_per_day(p_values, title="Dynamic Posterior Predictive Check"):
    plt.figure()
    plt.plot(range(1, len(p_values) + 1), p_values, marker="o")
    plt.axhline(0.1, linestyle="--")
    plt.xlabel("Time (day)")
    plt.ylabel("Predictive score")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def test1_temporal_masked_ppc(
    model_obj,
    A,
    X,
    obs_mask,
    num_z_samples=50,
    batch_size=128,
):
    t_obs, t_rep = ppc_discrepancy_per_user(
        model_obj=model_obj,
        A=A,
        X=X,
        obs_mask=obs_mask,
        num_z_samples=num_z_samples,
        batch_size=batch_size,
    )
    p_values = compute_predictive_scores_per_time(t_obs, t_rep)
    score_temporal = predictive_score(t_obs, t_rep)
    return {
        "t_obs": t_obs,
        "t_rep": t_rep,
        "ppc_per_day": p_values,
        "ppc_score_temporal": score_temporal,
    }


def test2_conditional_independence(
    model_obj,
    A,
    X,
    obs_mask,
    num_pairs=20,
    ridge_alpha=0.1,
):
    """
    Compare unconditional vs conditional correlations of causes given Z (and X).
    """
    device = next(model_obj.parameters()).device
    A = A.to(device)
    X = X.to(device)
    Z = infer_posterior_z(model_obj, A, X, obs_mask=obs_mask)

    A_np = A.detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    Z_np = Z.detach().cpu().numpy()

    U, T, D = A_np.shape
    flat_A = A_np.reshape(-1, D)
    flat_X = X_np.reshape(-1, X_np.shape[-1])
    flat_Z = Z_np.reshape(-1, Z_np.shape[-1])
    controls = np.hstack([flat_Z, flat_X])

    pairs = []
    rng = np.random.default_rng(0)
    while len(pairs) < min(num_pairs, D * (D - 1) // 2):
        i, j = rng.choice(D, 2, replace=False)
        if (i, j) not in pairs and (j, i) not in pairs:
            pairs.append((i, j))

    unconditional = []
    conditional = []
    for i, j in pairs:
        a_i = flat_A[:, i]
        a_j = flat_A[:, j]
        if a_i.std() == 0 or a_j.std() == 0:
            continue

        corr_unc = np.corrcoef(a_i, a_j)[0, 1]

        reg_i = Ridge(alpha=ridge_alpha).fit(controls, a_i)
        reg_j = Ridge(alpha=ridge_alpha).fit(controls, a_j)
        res_i = a_i - reg_i.predict(controls)
        res_j = a_j - reg_j.predict(controls)

        if res_i.std() == 0 or res_j.std() == 0:
            corr_cond = 0.0
        else:
            corr_cond = np.corrcoef(res_i, res_j)[0, 1]

        unconditional.append(abs(corr_unc))
        conditional.append(abs(corr_cond))

    reduction = float(np.mean(unconditional) - np.mean(conditional)) if unconditional else 0.0
    return {
        "pairs_tested": len(unconditional),
        "mean_abs_corr_uncond": float(np.mean(unconditional)) if unconditional else 0.0,
        "mean_abs_corr_cond": float(np.mean(conditional)) if conditional else 0.0,
        "corr_reduction": reduction,
    }


def test3_static_vs_temporal(t_obs, t_rep):
    """
    Compare static (time-collapsed) vs temporal PPC scores.
    """
    if isinstance(t_obs, np.ndarray):
        t_obs = torch.tensor(t_obs)
    if isinstance(t_rep, np.ndarray):
        t_rep = torch.tensor(t_rep)

    temporal_score = predictive_score(t_obs, t_rep)
    t_obs_sum = t_obs.sum(dim=1)        # (U,)
    t_rep_sum = t_rep.sum(dim=1)        # (U, S)
    static_score = predictive_score(t_obs_sum, t_rep_sum)

    return {
        "ppc_score_temporal": temporal_score,
        "ppc_score_static": static_score,
        "difference": float(temporal_score - static_score),
    }


def run_all_ppc_tests(
    model_obj,
    A,
    X,
    obs_mask,
    num_z_samples=50,
    batch_size=128,
    num_pairs=20,
) -> Dict:
    """
    Run the three PPC tests:
    1) Temporal masked PPC per day
    2) Conditional independence of causes given Z
    3) Static vs temporal comparison
    """
    test1 = test1_temporal_masked_ppc(
        model_obj=model_obj,
        A=A,
        X=X,
        obs_mask=obs_mask,
        num_z_samples=num_z_samples,
        batch_size=batch_size,
    )
    test2 = test2_conditional_independence(
        model_obj=model_obj,
        A=A,
        X=X,
        obs_mask=obs_mask,
        num_pairs=num_pairs,
    )
    test3 = test3_static_vs_temporal(
        t_obs=test1["t_obs"],
        t_rep=test1["t_rep"],
    )

    return {
        "test1_temporal_masked_ppc": test1,
        "test2_conditional_independence": test2,
        "test3_static_vs_temporal": test3,
    }
