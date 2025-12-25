import os, json, time
import torch
import pyro
import pyro.poutine
import pyro.distributions
import pyro.infer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde


def unwrap_distribution(fn):
    """
    Recursively unwrap Independent / MaskedDistribution
    to get the base distribution (e.g., Poisson, NegativeBinomial).
    """
    while hasattr(fn, "base_dist"):
        fn = fn.base_dist
    return fn


@torch.no_grad()
def ppc_discrepancy_per_user(
    model_obj,
    A,
    X,
    obs_mask,
    num_z_samples=30,
    batch_size=128,
):
    """
    Returns:
      t_obs: (U,)
      t_rep: (U, num_z_samples)
    """
    device = A.device
    model_obj.eval()

    U, T, D = A.shape
    held_mask = ~obs_mask  # True = held-out

    t_obs_all = []
    t_rep_all = []

    for start in range(0, U, batch_size):
        end = min(U, start + batch_size)

        A_b = A[start:end]
        X_b = X[start:end]
        M_b = obs_mask[start:end]
        H_b = held_mask[start:end]

        B = end - start

        # Sample latent sequences from guide
        predictive = pyro.infer.Predictive(
            model_obj.guide,
            num_samples=num_z_samples,
            return_sites=None,
        )
        guide_samples = predictive(A_b, X_b, obs_mask=M_b)

        z_names = sorted(
            [k for k in guide_samples if k.startswith("z_")],
            key=lambda s: int(s.split("_")[1])
        )

        t_obs_mc = torch.zeros(B, device=device)
        t_rep_mc = torch.zeros(B, num_z_samples, device=device)

        for s in range(num_z_samples):
            # Condition model on sampled z
            conditioned = {k: guide_samples[k][s] for k in z_names}
            conditioned_model = pyro.poutine.condition(model_obj.model, conditioned)

            # --- Run forward to get rates ---
            trace = pyro.poutine.trace(conditioned_model).get_trace(
                A_b, X_b, obs_mask=M_b, beta=1.0
            )

            # --- Compute log p(A | z) manually ---
            logp_obs = torch.zeros(B, device=device)
            logp_rep = torch.zeros(B, device=device)

            for t in range(T):
                # site = trace.nodes[f"A_{t+1}"]
                # rate = site["fn"].rate  # (B, D)
                site = trace.nodes[f"A_{t+1}"]
                fn = unwrap_distribution(site["fn"])

                rate = fn.rate   # (B, D)


                # log p(a | rate) per cause
                logp_full = torch.distributions.Poisson(rate).log_prob(A_b[:, t])

                # observed discrepancy (only held-out)
                logp_obs += (logp_full * H_b[:, t]).sum(dim=1)

                # replicate held-out causes
                A_rep_t = A_b[:, t].clone()
                A_rep_t[H_b[:, t]] = torch.distributions.Poisson(rate).sample()[H_b[:, t]]

                logp_rep_full = torch.distributions.Poisson(rate).log_prob(A_rep_t)
                logp_rep += (logp_rep_full * H_b[:, t]).sum(dim=1)

            t_obs_mc += logp_obs / num_z_samples
            t_rep_mc[:, s] = logp_rep

        t_obs_all.append(t_obs_mc.cpu())
        t_rep_all.append(t_rep_mc.cpu())

    t_obs = torch.cat(t_obs_all, dim=0)          # (U,)
    t_rep = torch.cat(t_rep_all, dim=0)          # (U, S)

    return t_obs, t_rep


@torch.no_grad()
def extract_posterior_mean_Z(model_obj, A, X, obs_mask):
    """
    Extract posterior mean of Z_{it} using the guide.
    Returns: Z_mean of shape (U, T, latent_dim)
    """
    model_obj.eval()

    predictive = pyro.infer.Predictive(
        model_obj.guide,
        num_samples=1,        # posterior mean, not sampling variability
        return_sites=None,
    )
    samples = predictive(A, X, obs_mask=obs_mask)

    z_names = sorted(
        [k for k in samples if k.startswith("z_")],
        key=lambda s: int(s.split("_")[1])
    )

    # Stack time dimension
    Z_mean = torch.stack([samples[k][0] for k in z_names], dim=1)
    # shape: (U, T, latent_dim)

    return Z_mean

def predictive_score(t_obs, t_rep):
  return (t_rep < t_obs[:, None]).float().mean().item()

def summarize_mask(obs_mask):
    """
    obs_mask: (U, T, D) bool
    Prints held-out ratio and per-user held-out counts.
    """
    U, T, D = obs_mask.shape
    held = (~obs_mask).float()
    held_ratio = held.mean().item()

    # If mask is fixed per-user across time, check consistency
    per_user_held = held[:, 0, :].sum(dim=1)  # (U,)
    print("Held-out ratio overall:", held_ratio)
    print("Held-out causes per user (min/mean/max):",
          per_user_held.min().item(),
          per_user_held.float().mean().item(),
          per_user_held.max().item())

    # Consistency check across time: held mask should be identical for all t
    inconsistent = (held != held[:, :1, :]).any().item()
    print("Mask inconsistent across time:", bool(inconsistent))


def save_run(
    save_dir,
    run_name,
    model_obj,
    A,
    X,
    obs_mask,
    train_df=None,
    train_kwargs=None,
    model_config=None,
    extra_meta=None,
    save_rng_state=True,
):
    """
    Save everything needed for:
    - PPC reproducibility
    - posterior Z reuse
    - downstream outcome estimation
    """
    os.makedirs(save_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"{run_name}_{stamp}"
    out = os.path.join(save_dir, tag)
    os.makedirs(out, exist_ok=True)

    # --------------------------------------------------
    # 1) Save Pyro param store (CRITICAL)
    # --------------------------------------------------
    pyro.get_param_store().save(os.path.join(out, "pyro_param_store.pt"))

    # --------------------------------------------------
    # 2) Save model state_dict
    # --------------------------------------------------
    torch.save(model_obj.state_dict(), os.path.join(out, "model_state_dict.pt"))

    # --------------------------------------------------
    # 3) Save obs_mask (CRITICAL for PPC)
    # --------------------------------------------------
    torch.save(obs_mask.cpu(), os.path.join(out, "obs_mask.pt"))

    # --------------------------------------------------
    # 4) Save posterior mean Z_{it} (CRITICAL for outcome model)
    # --------------------------------------------------
    Z_mean = extract_posterior_mean_Z(model_obj, A, X, obs_mask)
    torch.save(Z_mean.cpu(), os.path.join(out, "posterior_Z_mean.pt"))

    # --------------------------------------------------
    # 5) Save training logs
    # --------------------------------------------------
    if train_df is not None:
        train_df.to_csv(os.path.join(out, "train_logs.csv"), index=False)

    # --------------------------------------------------
    # 6) Save configs / metadata
    # --------------------------------------------------
    meta = {} if extra_meta is None else dict(extra_meta)
    meta.update({
        "run_name": run_name,
        "timestamp": stamp,
        "A_shape": list(A.shape),
        "X_shape": list(X.shape),
        "latent_dim": model_obj.latent_dim,
    })

    if train_kwargs is not None:
        with open(os.path.join(out, "train_kwargs.json"), "w") as f:
            json.dump(train_kwargs, f, indent=2)

    if model_config is not None:
        with open(os.path.join(out, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --------------------------------------------------
    # 7) Save RNG state (optional but good)
    # --------------------------------------------------
    if save_rng_state:
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
        }
        torch.save(rng_state, os.path.join(out, "rng_state.pt"))

    print("Run saved to:", out)
    return out



def plot_ppc_kde(
    t_obs,
    t_rep,
    ppc_score=None,
    title=None,
    figsize=(6, 4)
):
    """
    Paper-style PPC plot:
    - KDE of t_rep
    - Vertical dashed line at mean(t_obs)
    """

    # flatten replicated discrepancies
    t_rep_flat = t_rep.reshape(-1)

    obs_val = t_obs.mean().item()

    plt.figure(figsize=figsize)
    sns.kdeplot(
        t_rep_flat,
        fill=True,
        color="steelblue",
        alpha=0.6,
        label=r"$t(a^{rep}_{held})$"
    )

    plt.axvline(
        obs_val,
        color="black",
        linestyle="--",
        linewidth=2,
        label=r"$t(a_{held})$"
    )

    if ppc_score is not None:
        plt.text(
            0.02, 0.95,
            f"Predictive score = {ppc_score:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top"
        )

    plt.xlabel("log-likelihood")
    plt.ylabel("density")

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()



@torch.no_grad()
def ppc_discrepancy_time_indexed(
    model_obj,
    A,
    X,
    obs_mask,
    num_z_samples=30,
    batch_size=128,
):
    device = A.device
    model_obj.eval()

    U, T, D = A.shape
    held_mask = ~obs_mask

    t_obs_all = []
    t_rep_all = []

    for start in range(0, U, batch_size):
        end = min(U, start + batch_size)

        A_b = A[start:end]
        X_b = X[start:end]
        M_b = obs_mask[start:end]
        H_b = held_mask[start:end]

        B = end - start

        # sample Z
        predictive = pyro.infer.Predictive(
            model_obj.guide,
            num_samples=num_z_samples,
            return_sites=None,
        )
        guide_samples = predictive(A_b, X_b, obs_mask=M_b)

        z_names = sorted(
            [k for k in guide_samples if k.startswith("z_")],
            key=lambda s: int(s.split("_")[1])
        )

        t_obs_samples = torch.zeros(B, T, num_z_samples, device=device)
        t_rep_samples = torch.zeros(B, T, num_z_samples, device=device)

        for s in range(num_z_samples):
            conditioned = {k: guide_samples[k][s] for k in z_names}
            conditioned_model = pyro.poutine.condition(model_obj.model, conditioned)

            trace = pyro.poutine.trace(conditioned_model).get_trace(
                A_b, X_b, obs_mask=M_b, beta=1.0
            )

            for t in range(T):
                site = trace.nodes[f"A_{t+1}"]
                fn = unwrap_distribution(site["fn"])
                rate = fn.rate  # (B, D)

                # ----- observed discrepancy -----
                logp_obs = torch.distributions.Poisson(rate).log_prob(A_b[:, t])
                t_obs_samples[:, t, s] = (
                    logp_obs * H_b[:, t].float()
                ).mean(dim=1)   # ★ mean, not sum

                # ----- replicated discrepancy -----
                A_rep = torch.distributions.Poisson(rate).sample()
                logp_rep = torch.distributions.Poisson(rate).log_prob(A_rep)
                t_rep_samples[:, t, s] = (
                    logp_rep * H_b[:, t].float()
                ).mean(dim=1)

        t_obs_all.append(t_obs_samples.mean(dim=2).cpu())
        t_rep_all.append(t_rep_samples.cpu())

    t_obs = torch.cat(t_obs_all, dim=0)
    t_rep = torch.cat(t_rep_all, dim=0)

    return t_obs, t_rep



def compute_predictive_scores_per_time(t_obs, t_rep): 
  """ Compute predictive score for each time step. 
  Predictive score = P(t_rep < t_obs) Args: t_obs: (U, T) 
  - observed discrepancy t_rep: (U, T, S) 
  - replicated discrepancy 
  Returns: scores: (T,) 
  - predictive score at each time step 
  """ 
  U, T = t_obs.shape 
  S = t_rep.shape[2] 
  scores = torch.zeros(T) 

  for t in range(T): 
    # For each individual, compute P(t_rep < t_obs) 
    # Shape: (U, S) < (U, 1) -> (U, S) boolean 
    is_less = t_rep[:, t, :] < t_obs[:, t].unsqueeze(1)  # (U, S) # Average over samples and individuals 
    scores[t] = is_less.float().mean() 
  return scores


def compute_predictive_scores(t_obs, t_rep):
    """
    Args:
      t_obs: (U, T) - Observed discrepancy E_z[log p(a_held | z)]
      t_rep: (U, T, S) - Replicated discrepancies E_z[log p(a_rep | z)]
    
    Returns:
      scores: (T,) - The average predictive score for each time step
    """
    # 1. Compare each replicated sample to the observed mean
    # Result: (U, T, S) boolean tensor
    comparison = (t_rep < t_obs.unsqueeze(-1)).float()
    
    # 2. Calculate the score per individual per time step (average over S)
    # Result: (U, T)
    individual_scores = comparison.mean(dim=-1)
    
    # 3. Calculate the average score across the population for each time step
    # Result: (T,)
    time_indexed_scores = individual_scores.mean(dim=0)
    
    return time_indexed_scores



@torch.no_grad()
def ppc_discrepancy_time_indexed(
    model_obj,
    A,
    X,
    obs_mask,
    num_z_samples=30,
    batch_size=128,
):
    device = A.device
    model_obj.eval()

    U, T, D = A.shape
    held_mask = ~obs_mask

    t_obs_all = []
    t_rep_all = []

    for start in range(0, U, batch_size):
        end = min(U, start + batch_size)

        A_b = A[start:end]
        X_b = X[start:end]
        M_b = obs_mask[start:end]
        H_b = held_mask[start:end]

        B = end - start

        # sample Z
        predictive = pyro.infer.Predictive(
            model_obj.guide,
            num_samples=num_z_samples,
            return_sites=None,
        )
        guide_samples = predictive(A_b, X_b, obs_mask=M_b)

        z_names = sorted(
            [k for k in guide_samples if k.startswith("z_")],
            key=lambda s: int(s.split("_")[1])
        )

        t_obs_samples = torch.zeros(B, T, num_z_samples, device=device)
        t_rep_samples = torch.zeros(B, T, num_z_samples, device=device)

        for s in range(num_z_samples):
            conditioned = {k: guide_samples[k][s] for k in z_names}
            conditioned_model = pyro.poutine.condition(model_obj.model, conditioned)

            trace = pyro.poutine.trace(conditioned_model).get_trace(
                A_b, X_b, obs_mask=M_b, beta=1.0
            )

            for t in range(T):
                site = trace.nodes[f"A_{t+1}"]
                fn = unwrap_distribution(site["fn"])
                rate = fn.rate  # (B, D)

                # ----- observed discrepancy -----
                logp_obs = torch.distributions.Poisson(rate).log_prob(A_b[:, t])
                t_obs_samples[:, t, s] = (
                    logp_obs * H_b[:, t].float()
                ).mean(dim=1)   # ★ mean, not sum

                # ----- replicated discrepancy -----
                A_rep = torch.distributions.Poisson(rate).sample()
                logp_rep = torch.distributions.Poisson(rate).log_prob(A_rep)
                t_rep_samples[:, t, s] = (
                    logp_rep * H_b[:, t].float()
                ).mean(dim=1)

        t_obs_all.append(t_obs_samples.mean(dim=2).cpu())
        t_rep_all.append(t_rep_samples.cpu())

    t_obs = torch.cat(t_obs_all, dim=0)
    t_rep = torch.cat(t_rep_all, dim=0)

    return t_obs, t_rep


def plot_dynamic_ppc(t_obs, t_rep, threshold=0.1):
    # t_obs: (U,T), t_rep: (U,T,S)
    t_obs = np.asarray(t_obs)
    t_rep = np.asarray(t_rep)

    # PPC_t = P( t_rep < t_obs )
    ppc_t = (t_rep < t_obs[:, :, None]).mean(axis=(0, 2))  # (T,)

    plt.figure()
    plt.plot(np.arange(1, len(ppc_t) + 1), ppc_t, marker="o")
    plt.axhline(threshold, linestyle="--")
    plt.xlabel("Time (day)")
    plt.ylabel("Predictive score")
    plt.title("Dynamic Posterior Predictive Check")
    plt.tight_layout()
    plt.show()

    return ppc_t


def plot_posterior_and_prior_variance(post_var_t, prior_var_t):
    post_var_t = np.asarray(post_var_t)
    prior_var_t = np.asarray(prior_var_t)

    T = len(post_var_t)
    x = np.arange(1, T + 1)

    plt.figure()
    plt.plot(x, post_var_t, linewidth=2, label="Posterior variance")
    plt.plot(x, prior_var_t, linewidth=2, linestyle="--", label="Prior variance")

    plt.xlabel("Time (day)")
    plt.ylabel("Mean variance")
    plt.title("Posterior vs Prior Variance over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ppc_kde_per_day(
    t_obs,        # (U, T)
    t_rep,        # (U, T, S)
    scores,
    days=None,    # list of day indices
    figsize=(6, 4),
):
    """
    Day-wise PPC KDE consistent with PPC score:
    - Statistic: mean over users
    - KDE over replicated (u,s) samples
    - Vertical line at observed mean
    """

    U, T = t_obs.shape
    S = t_rep.shape[2]

    if days is None:
        days = list(range(T))

    for i in range(len(days)):
        t = days[i]
        # observed statistic (same as static)
        obs_val = t_obs[:, t].mean().item()

        # replicated statistics: flatten over users & posterior draws
        rep_vals = t_rep[:, t, :].reshape(-1).cpu().numpy()

        # PPC score (must match visual mass)
        score_t = scores[i]

        plt.figure(figsize=figsize)

        sns.kdeplot(
            rep_vals,
            fill=True,
            color="steelblue",
            alpha=0.6,
            label=r"$t_t(a^{rep}_{held})$"
        )

        plt.axvline(
            obs_val,
            color="black",
            linestyle="--",
            linewidth=2,
            label=r"$t_t(a_{held})$"
        )

        plt.text(
            0.02, 0.95,
            f"Predictive score = {score_t:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top"
        )

        plt.xlabel("log-likelihood")
        plt.ylabel("density")
        plt.title(f"Posterior Predictive Check (Day {t+1})")

        plt.legend()
        plt.tight_layout()
        plt.show()
