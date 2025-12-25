
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler


def build_purchase_tables(behaviors, binary = True, day_col="time_stamp"):
    """
    Returns:
      buy_day: DataFrame [user, day, y_buy] where y_buy is 0/1 (any purchase that day)
    """
    df = behaviors.copy()

    buy_day = (
        df.groupby(["user",day_col])["exposure_count"]
          .sum()
          .reset_index(name="buy_cnt")
    )
    if binary == True:
      buy_day["y_buy"] = (buy_day["buy_cnt"] > 0).astype(np.int8)
      buy_day = buy_day[["user",day_col,"y_buy"]]
    else:
      buy_day = buy_day[["user",day_col,"buy_cnt"]]
      buy_day["y_buy"] = buy_day["buy_cnt"]
      buy_day = buy_day[["user",day_col,"y_buy"]]
    return buy_day


def build_Y_tensors_from_buy_day(buy_day, user_ids, day_list):
    """
    buy_day: [user, day, y_buy]  (day is normalized timestamp)
    user_ids: length U (in same order as A_tensor axis0)
    day_list: length T (in same order as A_tensor axis1), normalized timestamps
    Returns:
      Y_day: (U,T) binary for same-day purchase
      Y_next: (U,T-1) binary for next-day purchase aligned with covariates at t (predict t+1)
      Y_cum: (U,T-1) binary for future cumulative purchase aligned with covariates at t (any in t+1..T-1)
    """
    U = len(user_ids); T = len(day_list)
    user2i = {u:i for i,u in enumerate(user_ids)}
    day2t  = {pd.Timestamp(d):t for t,d in enumerate(day_list)}

    Y_day = np.zeros((U,T), dtype=np.int8)
    for r in buy_day.itertuples(index=False):
        u, d, y = r.user, pd.Timestamp(r.date), r.y_buy
        if u in user2i and d in day2t:
            Y_day[user2i[u], day2t[d]] = 1

    # next-day (covariates at t -> outcome at t+1)
    Y_next = Y_day[:, 1:]                    # shape (U,T-1), corresponds to t=0..T-2
    # cumulative future (any purchase from t+1..end)
    # for each t in 0..T-2: Y_cum[:,t] = any(Y_day[:, t+1:])
    Y_cum = np.zeros((U, T-1), dtype=np.int8)
    future_any = np.zeros((U,), dtype=np.int8)
    # sweep from end backwards
    # At t = T-2, future window is only day T-1 => Y_day[:,T-1]
    future_any = Y_day[:, -1].copy()
    Y_cum[:, -1] = future_any
    for t in range(T-3, -1, -1):
        future_any = np.maximum(future_any, Y_day[:, t+1])
        Y_cum[:, t] = future_any

    return Y_day, Y_next, Y_cum




def plot_beta_distributions(betas_dict):
    plt.figure(figsize=(12, 6))
    for q, b in betas_dict.items():
        sns.kdeplot(b, label=f'Quantile {q}', fill=True, alpha=0.3)

    plt.axvline(0, color='red', linestyle='--', alpha=0.5)
    plt.title('Distribution of Causal Effects Across Different Purchase Quantiles')
    plt.xlabel('Marginal Effect (Beta)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def analyze_top_categories(betas_dict, A_mean, top_n=20):
    # Ensure A_mean is in numpy format
    if torch.is_tensor(A_mean):
        A_mean = A_mean.cpu().numpy()

    # Average beta across all quantiles as overall causal strength [cite: 515, 787]
    # This calculates the across-quantile average marginal effect
    avg_beta = np.mean(list(betas_dict.values()), axis=0)

    # Calculate total impact: average marginal effect * average exposure intensity [cite: 985]
    total_impact = avg_beta * A_mean

    df_impact = pd.DataFrame({
        'Category_ID': range(len(total_impact)),
        'Total_Impact': total_impact,
        'Avg_Beta': avg_beta,
        'A_mean': A_mean
    }).sort_values(by='Total_Impact', ascending=False)

    # Visualize Top N
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_impact.head(top_n), x='Total_Impact', y='Category_ID', orient='h', palette='magma')
    plt.title(f'Top {top_n} Ad Categories by Total Causal Impact (Weighted by Intensity)')
    plt.xlabel('Causal Impact (Average Beta * Mean Exposure)')
    plt.ylabel('Ad Category ID')
    plt.show()

    return df_impact


def estimate_outcome_model_fast(A, Z, X, Y, decoder_model):
    """
    Improved version: Estimate high-dimensional QTE using residualization + fast linear solver [cite: 565-566, 1106]
    """
    U, T, D_A = A.shape

    # --- Step 1: Reconstruct A_hat ---
    with torch.no_grad():
        A_hat_list = []
        for t in range(T):
            # Get expected exposure intensity at each time step [cite: 508-511, 1789]
            rate_t = F.softplus(decoder_model(torch.cat([Z[:, t], X[:, t]], dim=-1)))
            A_hat_list.append(rate_t)
        A_hat = torch.stack(A_hat_list, dim=1) # (U, T, D_A) [cite: 1819]

    # --- Step 2: Individual fixed effects processing (Demeaning) ---
    def demean_data(data):
        u_mean = data.mean(axis=1, keepdims=True)
        return (data - u_mean).reshape(-1, data.shape[-1])

    Y_demean = (Y.cpu().numpy() - Y.cpu().numpy().mean(axis=1, keepdims=True)).reshape(-1)
    A_demean = demean_data(A.cpu().numpy())
    A_hat_demean = demean_data(A_hat.cpu().numpy())
    X_demean = demean_data(X.cpu().numpy())

    # --- Step 3: Residualization ---
    # Construct control variable matrix [cite: 1104, 1108]
    controls = np.hstack([A_hat_demean, X_demean])

    # Fast Ridge preprocessing
    print("Residualizing Y and A...")
    y_reg = Ridge(alpha=1.0).fit(controls, Y_demean)
    Y_res = Y_demean - y_reg.predict(controls)

    # Batch debiasing for 2896 causes [cite: 572-574]
    a_reg = Ridge(alpha=1.0).fit(controls, A_demean)
    A_res = A_demean - a_reg.predict(controls)

    # --- Step 4: Fast quantile regression (SGD approximation) ---
    print(f"Estimating populatin via Fast SGD...")
    # SGD is extremely sensitive to feature scaling, must standardize
    scaler = StandardScaler()
    A_res_scaled = scaler.fit_transform(A_res)

    # Use SGDRegressor to implement high-dimensional quantile regression
    # loss='huber' with minimal epsilon excellently simulates quantile regression [cite: 572]
    model = SGDRegressor(
        loss='huber',
        epsilon=0.01,           # Keep minimal to approximate absolute deviation loss
        alpha=0.0001,          # L2 regularization to prevent overfitting [cite: 572]
        max_iter=1000,
        learning_rate='adaptive',
        eta0=0.01
    )

    model.fit(A_res_scaled, Y_res)

    # Restore coefficient units
    beta = model.coef_ / scaler.scale_

    return beta




def estimate_outcome_model_uncorrected(A, X, Y):
    """
    Compute uncorrected model (without using A_hat) for comparison analysis
    """
    # --- Step 1: Demeaning ---
    def demean_data(data):
        u_mean = data.mean(axis=1, keepdims=True)
        return (data - u_mean).reshape(-1, data.shape[-1])

    Y_demean = (Y.cpu().numpy() - Y.cpu().numpy().mean(axis=1, keepdims=True)).reshape(-1)
    A_demean = demean_data(A.cpu().numpy())
    X_demean = demean_data(X.cpu().numpy())

    # --- Step 2: Residualization (only using observed X) ---
    # Key difference here: removed A_hat_demean
    controls = X_demean

    y_reg = Ridge(alpha=1.0).fit(controls, Y_demean)
    Y_res = Y_demean - y_reg.predict(controls)

    a_reg = Ridge(alpha=1.0).fit(controls, A_demean)
    A_res = A_demean - a_reg.predict(controls)

    # --- Step 3: Weighted SGD quantile regression (same logic as before) ---
    scaler = StandardScaler()
    A_res_scaled = scaler.fit_transform(A_res)

    # Use SGDRegressor to implement high-dimensional quantile regression
    # loss='huber' with minimal epsilon excellently simulates quantile regression [cite: 572]
    model = SGDRegressor(
        loss='huber',
        epsilon=0.01,           # Keep minimal to approximate absolute deviation loss
        alpha=0.0001,          # L2 regularization to prevent overfitting [cite: 572]
        max_iter=1000,
        learning_rate='adaptive',
        eta0=0.01
    )

    model.fit(A_res_scaled, Y_res)

    # Restore coefficient units
    beta = model.coef_ / scaler.scale_

    return beta


def compare_deconfounding(beta_deconfounded, beta_uncorrected):
    plt.figure(figsize=(8, 8))
    plt.scatter(beta_uncorrected, beta_deconfounded, alpha=0.5, s=10)

    # Plot 45-degree reference line
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)

    plt.xlabel('Uncorrected Coefficients (Correlational)')
    plt.ylabel('Deconfounded Coefficients (Causal)')
    plt.title('Causal Correction: Deconfounded vs. Uncorrected Estimates')
    plt.grid(True)
    plt.show()

    # Calculate deviation ratio or Calculate bias proportion
    bias_reduction = np.mean(np.abs(beta_uncorrected - beta_deconfounded))
    print(f"Average coefficient adjustment: {bias_reduction}")




def estimate_temporal_population_uncorrected(A, X, Y):
    """
    Population-level average effects estimation with temporal structure (Uncorrected version - without Z)
    """
    U, T, D_A = A.shape

    # Convert inputs to NumPy format
    A_np, X_np, Y_np = A.cpu().numpy(), X.cpu().numpy(), Y.cpu().numpy()

    Y_res_3d = np.zeros((U, T))
    A_res_3d = np.zeros((U, T, D_A))

    print("Step 1: Performing daily residualization (Uncorrected - No Z)...")
    for t in range(T):
        # Key difference: control variables only include observed X, not A_hat
        controls = X_np[:, t, :]

        # Residualize each day independently, preserving temporal structure
        # Even without Z, remove the effect of X to improve estimation accuracy
        y_reg = Ridge(alpha=1.0).fit(controls, Y_np[:, t])
        Y_res_3d[:, t] = Y_np[:, t] - y_reg.predict(controls)

        a_reg = Ridge(alpha=1.0).fit(controls, A_np[:, t, :])
        A_res_3d[:, t, :] = A_np[:, t, :] - a_reg.predict(controls)

    # --- Step 2: Estimate instantaneous population effects (Instantaneous) ---
    A_inst = A_res_3d.reshape(-1, D_A)
    Y_inst = Y_res_3d.reshape(-1)
    beta_inst_unconf = Ridge(alpha=1.0).fit(A_inst, Y_inst).coef_

    # --- Step 3: Estimate cumulative population effects (Cumulative) ---
    A_cum = A_res_3d.sum(axis=1) # (12000, 2896)
    Y_cum = Y_res_3d.sum(axis=1) # (12000,)
    beta_cum_unconf = Ridge(alpha=1.0).fit(A_cum, Y_cum).coef_

    return beta_inst_unconf, beta_cum_unconf



def estimate_dual_causal_effects(A, Z, X, Y, decoder_model, quantile=0.95):
    U, T, D_A = A.shape

    # --- Stage 1: Daily dynamic debiasing ---
    with torch.no_grad():
        A_hat = torch.stack([F.softplus(decoder_model(torch.cat([Z[:, t], X[:, t]], dim=-1))) for t in range(T)], dim=1)

    A_np, A_hat_np, X_np, Y_np = A.cpu().numpy(), A_hat.cpu().numpy(), X.cpu().numpy(), Y.cpu().numpy()

    Y_res_3d = np.zeros((U, T))
    A_res_3d = np.zeros((U, T, D_A))

    for t in range(T):
        # Control variables for each day independently (no time pooling)
        controls = np.hstack([A_hat_np[:, t, :], X_np[:, t, :]])

        # Individual effects demeaning (Within-unit demeaning)
        y_demean = Y_np[:, t] - Y_np[:, t].mean()
        a_demean = A_np[:, t, :] - A_np[:, t, :].mean(axis=0)

        # Obtain residuals for current day
        Y_res_3d[:, t] = y_demean - Ridge(alpha=1.0).fit(controls, y_demean).predict(controls)
        A_res_3d[:, t, :] = a_demean - Ridge(alpha=1.0).fit(controls, a_demean).predict(controls)

    # --- Stage 2: Model A - Instantaneous effects ---
    A_inst = A_res_3d.reshape(-1, D_A)
    Y_inst = Y_res_3d.reshape(-1)
    beta_inst = fast_quantile_sgd(A_inst, Y_inst, quantile)

    # --- Stage 2: Model B - Cumulative effects ---
    A_cum = A_res_3d.sum(axis=1) # (12000, 2896)
    Y_cum = Y_res_3d.sum(axis=1) # (12000,)
    beta_cum = fast_quantile_sgd(A_cum, Y_cum, quantile)

    return beta_inst, beta_cum

def fast_quantile_sgd(X, y, q):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    # Get initial residual direction
    init_res = y - Ridge(alpha=1.0).fit(X_s, y).predict(X_s)
    weights = np.where(init_res > 0, q, 1 - q)
    # Weighted SGD to simulate quantile regression
    model = SGDRegressor(loss='epsilon_insensitive', epsilon=0, max_iter=2000)
    model.fit(X_s, y, sample_weight=weights)
    return model.coef_ / scaler.scale_




def estimate_temporal_population_effects(A, Z, X, Y, decoder_model):
    """
    Population-level average effects estimation with temporal structure (Population Level)
    """
    U, T, D_A = A.shape

    # --- Step 1: Daily debiasing to obtain residuals (no time pooling) ---
    with torch.no_grad():
        # Generate A_hat for each day (proxy for Z)
        A_hat = torch.stack([F.softplus(decoder_model(torch.cat([Z[:, t], X[:, t]], dim=-1)))
                           for t in range(T)], dim=1).cpu().numpy()

    A_np, X_np, Y_np = A.cpu().numpy(), X.cpu().numpy(), Y.cpu().numpy()
    Y_res_3d = np.zeros((U, T))
    A_res_3d = np.zeros((U, T, D_A))

    print("Step 1: Performing daily deconfounding...")
    for t in range(T):
        # Construct confounder control matrix for current day
        controls = np.hstack([A_hat[:, t, :], X_np[:, t, :]])

        # Debias each day independently, preserving temporal structure
        y_reg = Ridge(alpha=1.0).fit(controls, Y_np[:, t])
        Y_res_3d[:, t] = Y_np[:, t] - y_reg.predict(controls)

        a_reg = Ridge(alpha=1.0).fit(controls, A_np[:, t, :])
        A_res_3d[:, t, :] = A_np[:, t, :] - a_reg.predict(controls)

    # --- Step 2: Estimate instantaneous population effects (Instantaneous Population Beta) ---
    print("Step 2: Estimating Instantaneous Effects...")
    # Stack all time steps
    A_inst = A_res_3d.reshape(-1, D_A)
    Y_inst = Y_res_3d.reshape(-1)
    # Linear regression to obtain average effects
    beta_inst_pop = Ridge(alpha=1.0).fit(A_inst, Y_inst).coef_

    # --- Step 3: Estimate cumulative population effects (Cumulative Population Beta) ---
    print("Step 3: Estimating Cumulative Effects...")
    # Sum over 9 days at user level
    A_cum = A_res_3d.sum(axis=1)
    Y_cum = Y_res_3d.sum(axis=1)
    # Linear regression to obtain cumulative average effects
    beta_cum_pop = Ridge(alpha=1.0).fit(A_cum, Y_cum).coef_

    return beta_inst_pop, beta_cum_pop