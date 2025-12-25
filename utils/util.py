


import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.poutine as poutine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial



def _get_elbo_components(model_obj, A_batch, X_batch):
    """
    Returns: (elbo, recon_nll, kl, posterior_var_mean)
    elbo here is negative ELBO loss value (higher is better); we also return the standard SVI loss.
    """
    # Trace guide then replay model
    guide_trace = poutine.trace(model_obj.guide).get_trace(A_batch, X_batch)
    guide_trace.compute_log_prob()   # 🔴 必须
    
    model_trace = poutine.trace(poutine.replay(model_obj.model, trace=guide_trace)).get_trace(A_batch, X_batch)
    model_trace.compute_log_prob()


    # Compute log probs
    model_logp = 0.0
    guide_logp = 0.0
    recon_logp = 0.0

    # Identify latent sites by name pattern
    latent_sites = []
    for name, site in guide_trace.nodes.items():
        if site.get("type") == "sample" and (name.startswith("z_") or name.startswith("latent") or name.startswith("z")):
            if not site.get("is_observed", False):
                latent_sites.append(name)

    # Sum model log prob over all sample sites, and separate observed likelihood (recon)
    for name, site in model_trace.nodes.items():
        if site.get("type") == "sample":
            logp = site["log_prob"].sum()
            model_logp = model_logp + logp
            if site.get("is_observed", False):
                recon_logp = recon_logp + logp

    # Sum guide log prob over latent sample sites
    for name, site in guide_trace.nodes.items():
        if site.get("type") == "sample":
            if (not site.get("is_observed", False)) and (name in latent_sites):
                guide_logp = guide_logp + site["log_prob"].sum()

    # ELBO = E_q [log p - log q]
    elbo = (model_logp - guide_logp).detach().item()

    # recon_nll = -log p(A | z, X)
    recon_nll = (-recon_logp).detach().item()

    # KL = E_q [log q - log p(z)]  (computed implicitly)
    kl = (guide_logp - (model_logp - recon_logp)).detach().item()
    
    # Explanation: model_logp = log p(z) + log p(A|z, X)
    # so model_logp - recon_logp = log p(z)

    # Posterior variance mean from guide distributions of latent sites
    vars_ = []
    for name in latent_sites:
        site = guide_trace.nodes[name]
        fn = site["fn"]
        # Normal(loc, scale).to_event(...)
        if hasattr(fn, "base_dist"):
            base = fn.base_dist
        else:
            base = fn
        if hasattr(base, "scale"):
            var = (base.scale ** 2).detach()
            vars_.append(var.mean().item())
    post_var_mean = float(np.mean(vars_)) if len(vars_) > 0 else np.nan

    return elbo, recon_nll, kl, post_var_mean




########################################################
########################################################



def linear_anneal(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch + 1) / warmup_epochs))


def cyclical_anneal(epoch, cycle_length, ratio=0.5):
    """
    cycle_length: total epochs per cycle
    ratio: fraction of cycle spent increasing from 0->1, remainder stays at 1
    """
    if cycle_length <= 0:
        return 1.0
    pos = (epoch % cycle_length) / cycle_length
    if pos < ratio:
        return float(pos / ratio)
    return 1.0



def train_dvae_with_diagnostics(
    model_obj,
    A_tensor,
    X_tensor,
    num_epochs=200,
    batch_size=64,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    anneal="linear",              # "linear" or "cyclical" or "none"
    warmup_epochs=50,             # for linear
    cycle_length=50,              # for cyclical
    cycle_ratio=0.5,
    seed=0,
):
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    model_obj.to(device)
    A_tensor = A_tensor.to(device)
    X_tensor = X_tensor.to(device)

    optimizer = Adam({"lr": lr})
    base_elbo = Trace_ELBO()

    n = A_tensor.shape[0]
    logs = []

    for epoch in range(num_epochs):
        # anneal factor beta
        if anneal == "linear":
            beta = linear_anneal(epoch, warmup_epochs)
        elif anneal == "cyclical":
            beta = cyclical_anneal(epoch, cycle_length, cycle_ratio)
        else:
            beta = 1.0

        def model_with_beta(A, X):
            return model_obj.model(A, X, beta=beta)
    
        svi = SVI(model_with_beta, model_obj.guide, optimizer, loss=base_elbo)
    
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        epoch_elbo = 0.0
        epoch_kl = 0.0
        epoch_recon = 0.0
        epoch_var = []

        num_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            A_b = A_tensor[idx]
            X_b = X_tensor[idx]

            loss = svi.step(A_b, X_b)
            epoch_loss += loss
            num_batches += 1

            # Diagnostics on-the-fly
            with torch.no_grad():  # ← 添加no_grad以提高效率
                elbo_val, recon_nll, kl_val, post_var = _get_elbo_components(model_obj, A_b, X_b)
            epoch_elbo += elbo_val
            epoch_recon += recon_nll
            epoch_kl += kl_val
            epoch_var.append(post_var)

        # Average per batch
        avg_loss = epoch_loss / num_batches
        avg_elbo = epoch_elbo / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        avg_var = float(np.nanmean(epoch_var))

        logs.append({
            "epoch": epoch + 1,
            "beta": beta,
            "svi_loss": avg_loss,       # = -ELBO up to Monte Carlo noise
            "elbo": avg_elbo,
            "recon_nll": avg_recon,
            "kl": avg_kl,
            "kl_weighted": beta * avg_kl,     # ← 实际被惩罚的 KL
            "post_var_mean": avg_var,
        })

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] beta={beta:.3f} loss={avg_loss:.2f} KL={avg_kl:.2f} var={avg_var:.4f}")

    df = pd.DataFrame(logs)

    # ---- Plots ----
    plt.figure()
    plt.plot(df["epoch"], df["kl"])
    plt.xlabel("epoch")
    plt.ylabel("KL")
    plt.title("KL over training")
    plt.show()

    plt.figure()
    plt.plot(df["epoch"], df["post_var_mean"])
    plt.xlabel("epoch")
    plt.ylabel("Posterior variance mean")
    plt.title("Posterior variance over training")
    plt.show()

    return df

def make_fixed_user_mask(U, T, D, holdout_frac=0.05, seed=0, device="cuda"):
    """
    Create a fixed per-user mask over causes.

    obs_mask[u, t, d] = True  -> observed (included in ELBO)
    obs_mask[u, t, d] = False -> held-out (excluded from ELBO, used for PPC)
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    k = max(1, int(round(D * holdout_frac)))
    held_user = torch.zeros(U, D, dtype=torch.bool)

    for u in range(U):
        idx = torch.randperm(D, generator=g)[:k]
        held_user[u, idx] = True

    obs_user = ~held_user
    obs_mask = obs_user[:, None, :].expand(U, T, D).contiguous().to(device)

    return obs_mask



def make_time_varying_user_mask(U, T, D, holdout_frac=0.05, seed=0, device="cuda"):
    """
    PPC-B: per-user, per-time random holdout of causes
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    k = max(1, int(round(D * holdout_frac)))
    obs_mask = torch.ones(U, T, D, dtype=torch.bool)

    for u in range(U):
        for t in range(T):
            idx = torch.randperm(D, generator=g)[:k]
            obs_mask[u, t, idx] = False

    return obs_mask.to(device)


