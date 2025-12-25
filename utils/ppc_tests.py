"""
Time-Series Deconfounder: Posterior Predictive Checking (PPC)

根据 "Blessings of Multiple Causes" (Wang & Blei, 2019) 改编为时序设置

Key idea:
  如果Z是good substitute confounder，那么：
  1. 能准确预测held-out data
  2. Causes conditional on Z应该独立
  3. 能准确预测未来时间步
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss
import pyro
import pyro.poutine as poutine


# ============================================================
# Test 1: Held-out Predictive Likelihood (最重要)
# ============================================================

def test1_heldout_predictive_likelihood(
    model_obj, 
    A_train, X_train,      # Training data
    A_held, X_held,        # Held-out data
    num_samples=100
):
    """
    测试模型在held-out set上的预测能力
    
    Compare:
      1. No control: p(A_held | X_held)
      2. Control for Z: p(A_held | Z, X_held) ← 你的模型
      3. Control for Z + filtering: 用training data的Z
    
    Returns:
      - log_likelihoods: dict of average log p(A_held)
      - predictions: 预测的A_held
    """
    print("\n" + "="*70)
    print(" TEST 1: Held-out Predictive Likelihood")
    print("="*70)
    
    model_obj.eval()
    device = next(model_obj.parameters()).device
    A_held_t = A_held.to(device)
    X_held_t = X_held.to(device)
    
    results = {}
    
    # -------- Baseline 1: No Control --------
    print("\n1. Baseline (No Control): p(A | X)")
    print("   用简单模型（不考虑Z）预测")
    
    # 简单方法：用training data的empirical mean
    A_train_mean = A_train.mean(axis=0)  # [T, n_categories]
    
    # 预测held-out
    with torch.no_grad():
        # Poisson log likelihood
        A_train_mean_t = torch.FloatTensor(A_train_mean).to(device)
        rate_baseline = A_train_mean_t.unsqueeze(0).expand_as(A_held_t) + 1e-6
        
        # Log p(A_held | rate)
        ll_baseline = (A_held_t * torch.log(rate_baseline) - rate_baseline).sum(dim=(1, 2))
        avg_ll_baseline = ll_baseline.mean().item()
    
    results['no_control'] = avg_ll_baseline
    print(f"   Avg Log Likelihood: {avg_ll_baseline:.2f}")
    
    
    # -------- Method 2: Control for Z (你的模型) --------
    print("\n2. Control for Z: p(A | Z, X)")
    print("   用你的Dynamic VAE（filtering guide）")
    
    with torch.no_grad():
        # 用guide推断Z
        guide_trace = poutine.trace(model_obj.guide).get_trace(A_held_t, X_held_t)
        
        # 提取所有z_t
        Z_posterior = []
        for t in range(A_held.shape[1]):
            z_name = f"z_{t+1}"
            if z_name in guide_trace.nodes:
                Z_posterior.append(guide_trace.nodes[z_name]["value"])
        
        # 用model计算p(A | Z, X)
        # 重新sample多次计算期望
        log_likelihoods = []
        
        for _ in range(num_samples):
            # 用model生成，但observe A
            model_trace = poutine.trace(model_obj.model).get_trace(A_held_t, X_held_t)
            model_trace.compute_log_prob()
            
            # 提取观测的log prob
            total_ll = 0.0
            for name, site in model_trace.nodes.items():
                if site.get("is_observed", False) and name.startswith("A_"):
                    total_ll += site["log_prob"].sum()
            
            log_likelihoods.append(total_ll.item())
        
        avg_ll_control = np.mean(log_likelihoods)
    
    results['control_z'] = avg_ll_control
    print(f"   Avg Log Likelihood: {avg_ll_control:.2f}")
    
    
    # -------- Method 3: Sequential Filtering --------
    print("\n3. Sequential Filtering: p(A_held | Z_train, X)")
    print("   用training data推断的Z预测held-out")
    
    # 这个需要你的model支持：先用training data得到Z，再预测held-out
    # 这是时序模型特有的测试
    
    
    # -------- 对比 --------
    print("\n" + "="*70)
    print(" COMPARISON")
    print("="*70)
    print(f"{'Model':<30} {'Avg Log-Likelihood':>20} {'Improvement':>15}")
    print("-"*70)
    
    baseline = results['no_control']
    for name, ll in results.items():
        improvement = ll - baseline
        print(f"{name:<30} {ll:>20.2f} {improvement:>15.2f}")
    
    print("-"*70)
    
    if results['control_z'] > results['no_control']:
        print("\n✅ Control for Z improves prediction!")
        print("   → Z captures useful information about confounders")
    else:
        print("\n⚠️  Control for Z does NOT improve prediction")
        print("   → Z may not be a good substitute confounder")
    
    return results


# ============================================================
# Test 2: Leave-One-Cause-Out (LOCO)
# ============================================================

def test2_leave_one_cause_out(
    model_obj,
    A_train, X_train,
    num_test_causes=20,
    num_samples=50
):
    """
    测试causes是否conditional on Z独立
    
    Idea:
      如果Z是good substitute:
        p(A_j | A_{-j}, Z, X) ≈ p(A_j | Z, X)
        (第j个cause只依赖Z，不依赖其他causes)
    
    Procedure:
      1. For each cause j (category):
         - Mask A_j (设为0)
         - 用A_{-j}推断Z
         - 预测A_j: p(A_j | Z, X)
         - 比较：预测 vs 真实
    
    Compare:
      - Model 1: p(A_j | Z, X)           ← Control for Z
      - Model 2: p(A_j | A_{-j}, X)      ← Direct dependency (坏的!)
      - Model 3: p(A_j | X)              ← No control
    
    Returns:
      - scores: dict of {category: {model: score}}
    """
    print("\n" + "="*70)
    print(" TEST 2: Leave-One-Cause-Out (LOCO)")
    print("="*70)
    print("\nIdea: 如果Z是good substitute, A_j | Z, X 应该独立于 A_{-j}")
    
    model_obj.eval()
    device = next(model_obj.parameters()).device
    
    n_users, T, n_categories = A_train.shape
    
    # 随机选择要测试的categories
    if num_test_causes > n_categories:
        num_test_causes = n_categories
    
    test_categories = np.random.choice(n_categories, num_test_causes, replace=False)
    
    results = {
        'category': [],
        'sparsity': [],
        'll_with_z': [],          # p(A_j | Z, X)
        'll_without_z': [],       # p(A_j | X)
        'baseline_mean': []       # 简单baseline
    }
    
    for cat_idx in test_categories:
        print(f"\nTesting category {cat_idx}...")
        
        # 真实的A_j
        A_j_true = A_train[:, :, cat_idx]  # [n_users, T]
        sparsity = (A_j_true == 0).mean()
        
        if A_j_true.sum() < 10:  # 跳过太稀疏的
            print(f"  Skipped (too sparse: {sparsity:.1%})")
            continue
        
        # -------- Mask A_j --------
        A_masked = A_train.copy()
        A_masked[:, :, cat_idx] = 0  # 去掉第j个cause
        
        A_masked_t = torch.FloatTensor(A_masked).to(device)
        X_train_t = torch.FloatTensor(X_train).to(device)
        A_j_true_t = torch.FloatTensor(A_j_true).to(device)
        
        # -------- Model 1: Control for Z --------
        with torch.no_grad():
            # 用A_{-j}推断Z
            guide_trace = poutine.trace(model_obj.guide).get_trace(A_masked_t, X_train_t)
            
            # 提取Z
            Z_inferred = []
            for t in range(T):
                z_name = f"z_{t+1}"
                if z_name in guide_trace.nodes:
                    Z_inferred.append(guide_trace.nodes[z_name]["value"])
            
            # 用Z和X预测A_j
            # 这里需要一个prediction network: p(A_j | Z, X)
            # 简化：用emission network
            predicted_rates = []
            for t in range(T):
                z_t = Z_inferred[t]
                x_t = X_train_t[:, t, :]
                
                # 用emission预测
                rate = torch.nn.functional.softplus(
                    model_obj.emission(torch.cat([z_t, x_t], dim=-1))
                )
                predicted_rates.append(rate[:, cat_idx])  # 第j个category的rate
            
            predicted_rates = torch.stack(predicted_rates, dim=1)  # [n_users, T]
            
            # 计算log likelihood
            ll_with_z = (
                A_j_true_t * torch.log(predicted_rates + 1e-8) - predicted_rates
            ).mean().item()
        
        # -------- Baseline: No Control --------
        # 简单预测：用training mean
        baseline_rate = A_j_true.mean() + 1e-6
        ll_baseline = (
            A_j_true * np.log(baseline_rate) - baseline_rate
        ).mean()
        
        # 记录结果
        results['category'].append(cat_idx)
        results['sparsity'].append(sparsity)
        results['ll_with_z'].append(ll_with_z)
        results['ll_without_z'].append(ll_baseline)
        results['baseline_mean'].append(baseline_rate)
        
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  LL with Z: {ll_with_z:.3f}")
        print(f"  LL baseline: {ll_baseline:.3f}")
        print(f"  Improvement: {ll_with_z - ll_baseline:.3f}")
    
    # -------- Summary --------
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"\nTested {len(results['category'])} categories")
    print(f"Average LL with Z: {np.mean(results['ll_with_z']):.3f}")
    print(f"Average LL baseline: {np.mean(results['ll_without_z']):.3f}")
    print(f"Average Improvement: {np.mean(results['ll_with_z']) - np.mean(results['ll_without_z']):.3f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df_results['ll_without_z'], df_results['ll_with_z'], alpha=0.6)
    plt.plot([-10, 0], [-10, 0], 'r--', label='y=x')
    plt.xlabel('LL without Z (baseline)')
    plt.ylabel('LL with Z (deconfounder)')
    plt.title('LOCO: Predictive Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    improvements = df_results['ll_with_z'] - df_results['ll_without_z']
    plt.hist(improvements, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Improvement (LL with Z - LL baseline)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Improvements')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test2_loco_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 统计检验
    t_stat, p_val = stats.ttest_1samp(improvements, 0)
    print(f"\nStatistical Test:")
    print(f"  H0: No improvement (mean = 0)")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_val:.4f}")
    
    if p_val < 0.05 and t_stat > 0:
        print("  ✅ Significant improvement! Z is useful.")
    else:
        print("  ⚠️  No significant improvement. Z may not be good substitute.")
    
    return df_results


# ============================================================
# Test 3: Temporal Forecasting
# ============================================================

def test3_temporal_forecasting(
    model_obj,
    A_train, X_train,      # Day 1-7
    A_held, X_held,        # Day 8-9
    forecast_steps=[1, 2]  # 预测1步和2步
):
    """
    测试时序预测能力
    
    Procedure:
      1. 用Day 1-7训练，推断Z_{1:7}
      2. 预测Day 8: p(A_8 | Z_{1:7}, A_{1:7}, X_{1:8})
      3. 预测Day 9: p(A_9 | Z_{1:8}, A_{1:8}, X_{1:9})
    
    Compare:
      - Dynamic model (你的): 考虑时序
      - Static baseline: 用历史均值
      - Last-day baseline: 用最后一天的值
    
    Returns:
      - forecast_results: dict of metrics per step
    """
    print("\n" + "="*70)
    print(" TEST 3: Temporal Forecasting")
    print("="*70)
    
    model_obj.eval()
    device = next(model_obj.parameters()).device
    
    n_users, T_train, n_categories = A_train.shape
    T_held = A_held.shape[1]
    
    results = {
        'step': [],
        'll_dynamic': [],       # 你的dynamic model
        'll_static': [],        # 历史均值
        'll_lastday': [],       # 最后一天
        'mse_dynamic': [],
        'mse_static': [],
        'mse_lastday': []
    }
    
    # Convert to tensors
    A_train_t = torch.FloatTensor(A_train).to(device)
    X_train_t = torch.FloatTensor(X_train).to(device)
    A_held_t = torch.FloatTensor(A_held).to(device)
    X_held_t = torch.FloatTensor(X_held).to(device)
    
    for step in forecast_steps:
        if step > T_held:
            continue
        
        print(f"\n--- Forecasting {step}-step ahead (Day {T_train + step}) ---")
        
        # 真实值
        A_target = A_held[:, step-1, :]  # [n_users, n_categories]
        A_target_t = A_held_t[:, step-1, :]
        
        # -------- Dynamic Model (你的) --------
        with torch.no_grad():
            # 用training data推断Z_{1:T_train}
            guide_trace = poutine.trace(model_obj.guide).get_trace(A_train_t, X_train_t)
            
            # 获取最后一个z
            z_last = guide_trace.nodes[f"z_{T_train}"]["value"]
            
            # 用transition预测z_{T+step}
            # 简化：假设A和X保持不变（或用held-out的）
            if step == 1:
                # 预测Day 8
                trans_in = torch.cat([
                    z_last,
                    A_train_t[:, -1, :],  # Last day of training
                    X_train_t[:, -1, :]
                ], dim=-1)
                mu_next, logvar_next = model_obj.transition(trans_in).chunk(2, dim=-1)
                z_next = mu_next  # 用mean作为point estimate
            else:
                # Multi-step需要递归预测（简化）
                z_next = z_last
            
            # 用z_next和X预测A
            x_target = X_held_t[:, step-1, :]
            rate_dynamic = torch.nn.functional.softplus(
                model_obj.emission(torch.cat([z_next, x_target], dim=-1))
            )
            rate_dynamic = torch.clamp(rate_dynamic, min=1e-6, max=1e3)
            
            # Log likelihood
            ll_dynamic = (
                A_target_t * torch.log(rate_dynamic + 1e-8) - rate_dynamic
            ).sum(dim=1).mean().item()
            
            # MSE
            mse_dynamic = ((A_target_t - rate_dynamic) ** 2).mean().item()
        
        # -------- Static Baseline --------
        # 用training data的时序平均
        rate_static = A_train.mean(axis=(0, 1)) + 1e-6  # [n_categories]
        ll_static = (
            A_target * np.log(rate_static) - rate_static
        ).sum(axis=1).mean()
        mse_static = ((A_target - rate_static) ** 2).mean()
        
        # -------- Last-day Baseline --------
        # 用最后一天的值
        rate_lastday = A_train[:, -1, :].mean(axis=0) + 1e-6  # [n_categories]
        ll_lastday = (
            A_target * np.log(rate_lastday) - rate_lastday
        ).sum(axis=1).mean()
        mse_lastday = ((A_target - rate_lastday) ** 2).mean()
        
        # 记录
        results['step'].append(step)
        results['ll_dynamic'].append(ll_dynamic)
        results['ll_static'].append(ll_static)
        results['ll_lastday'].append(ll_lastday)
        results['mse_dynamic'].append(mse_dynamic)
        results['mse_static'].append(mse_static)
        results['mse_lastday'].append(mse_lastday)
        
        print(f"  LL - Dynamic: {ll_dynamic:.3f}")
        print(f"  LL - Static:  {ll_static:.3f}")
        print(f"  LL - Lastday: {ll_lastday:.3f}")
        print(f"  MSE - Dynamic: {mse_dynamic:.3f}")
        print(f"  MSE - Static:  {mse_static:.3f}")
    
    # 可视化
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Log Likelihood
    ax = axes[0]
    x = df_results['step']
    ax.plot(x, df_results['ll_dynamic'], marker='o', linewidth=2, 
            markersize=10, label='Dynamic (with Z)', color='#2E86AB')
    ax.plot(x, df_results['ll_static'], marker='s', linewidth=2,
            markersize=10, label='Static baseline', color='#F18F01')
    ax.plot(x, df_results['ll_lastday'], marker='^', linewidth=2,
            markersize=10, label='Last-day baseline', color='#A23B72')
    ax.set_xlabel('Forecast Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Log-Likelihood', fontsize=12, fontweight='bold')
    ax.set_title('Temporal Forecasting Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MSE
    ax = axes[1]
    ax.plot(x, df_results['mse_dynamic'], marker='o', linewidth=2,
            markersize=10, label='Dynamic (with Z)', color='#2E86AB')
    ax.plot(x, df_results['mse_static'], marker='s', linewidth=2,
            markersize=10, label='Static baseline', color='#F18F01')
    ax.plot(x, df_results['mse_lastday'], marker='^', linewidth=2,
            markersize=10, label='Last-day baseline', color='#A23B72')
    ax.set_xlabel('Forecast Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test3_temporal_forecasting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print(" TEMPORAL FORECASTING SUMMARY")
    print("="*70)
    
    avg_improvement = (
        np.mean(df_results['ll_dynamic']) - np.mean(df_results['ll_static'])
    )
    
    print(f"\nAverage improvement over static baseline: {avg_improvement:.3f}")
    
    if avg_improvement > 0:
        print("✅ Dynamic model (with Z) outperforms static baseline!")
        print("   → Temporal structure in Z is useful")
    else:
        print("⚠️  Dynamic model does not improve over baseline")
        print("   → Z may not capture temporal dynamics")
    
    return df_results


# ============================================================
# Test 4: Substitute Confounder Validity
# ============================================================

def test4_substitute_confounder_validity(
    model_obj,
    A_train, X_train,
    A_held, X_held
):
    """
    测试Z是否满足substitute confounder的条件
    
    根据Blessings of Causes, substitute confounder需要满足：
    1. Z能explain causes的dependence
    2. Conditional on Z, causes应该接近独立
    3. Z能预测outcome
    
    Tests:
      a) Pairwise independence test
      b) Conditional correlation
      c) Z的信息量
    """
    print("\n" + "="*70)
    print(" TEST 4: Substitute Confounder Validity")
    print("="*70)
    
    model_obj.eval()
    device = next(model_obj.parameters()).device
    
    A_train_t = torch.FloatTensor(A_train).to(device)
    X_train_t = torch.FloatTensor(X_train).to(device)
    
    with torch.no_grad():
        # 推断Z
        guide_trace = poutine.trace(model_obj.guide).get_trace(A_train_t, X_train_t)
        
        # 提取所有Z
        Z_all = []
        for t in range(A_train.shape[1]):
            z_name = f"z_{t+1}"
            if z_name in guide_trace.nodes:
                Z_all.append(guide_trace.nodes[z_name]["value"].cpu().numpy())
        
        Z_all = np.stack(Z_all, axis=1)  # [n_users, T, latent_dim]
    
    # -------- Test a: Pairwise Cause Correlation --------
    print("\na) Pairwise Cause Correlation")
    print("   测试causes之间的correlation是否被Z解释")
    
    # 随机选择10对categories
    n_users, T, n_categories = A_train.shape
    n_pairs = min(10, n_categories * (n_categories - 1) // 2)
    
    pairs = []
    while len(pairs) < n_pairs:
        i, j = np.random.choice(n_categories, 2, replace=False)
        if (i, j) not in pairs and (j, i) not in pairs:
            pairs.append((i, j))
    
    unconditional_corrs = []
    conditional_corrs = []
    
    for i, j in pairs:
        # Flatten across users and time
        A_i = A_train[:, :, i].flatten()
        A_j = A_train[:, :, j].flatten()
        Z_flat = Z_all.reshape(-1, Z_all.shape[-1])
        
        # Unconditional correlation
        if A_i.std() > 0 and A_j.std() > 0:
            corr_unc = np.corrcoef(A_i, A_j)[0, 1]
        else:
            continue
        
        # Conditional correlation (partial correlation given Z)
        # 简化：用residual correlation
        from sklearn.linear_model import Ridge
        
        # Regress A_i on Z
        reg_i = Ridge(alpha=0.1)
        reg_i.fit(Z_flat, A_i)
        residual_i = A_i - reg_i.predict(Z_flat)
        
        # Regress A_j on Z
        reg_j = Ridge(alpha=0.1)
        reg_j.fit(Z_flat, A_j)
        residual_j = A_j - reg_j.predict(Z_flat)
        
        # Correlation of residuals
        if residual_i.std() > 0 and residual_j.std() > 0:
            corr_cond = np.corrcoef(residual_i, residual_j)[0, 1]
        else:
            corr_cond = 0.0
        
        unconditional_corrs.append(abs(corr_unc))
        conditional_corrs.append(abs(corr_cond))
    
    print(f"   Avg |Correlation| without Z: {np.mean(unconditional_corrs):.3f}")
    print(f"   Avg |Correlation| with Z:    {np.mean(conditional_corrs):.3f}")
    print(f"   Reduction: {np.mean(unconditional_corrs) - np.mean(conditional_corrs):.3f}")
    
    if np.mean(conditional_corrs) < np.mean(unconditional_corrs):
        print("   ✅ Z reduces correlation → Z captures confounding!")
    else:
        print("   ⚠️  Z does not reduce correlation")
    
    # -------- Test b: Z的信息量 --------
    print("\nb) Latent Variable Information")
    print("   检查Z的variance是否充分（不是point mass）")
    
    Z_var = Z_all.var(axis=(0, 1))  # 每个latent dimension的variance
    print(f"   Z variance per dimension: {Z_var}")
    print(f"   Mean variance: {Z_var.mean():.4f}")
    
    if Z_var.mean() < 0.01:
        print("   ⚠️  Z variance collapsed → Z is nearly deterministic!")
    elif Z_var.mean() > 0.95:
        print("   ⚠️  Z variance near prior → Z not learning!")
    else:
        print("   ✅ Z variance healthy")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Correlation comparison
    ax = axes[0]
    ax.scatter(unconditional_corrs, conditional_corrs, alpha=0.6, s=100)
    ax.plot([0, max(unconditional_corrs)], [0, max(unconditional_corrs)], 
            'r--', linewidth=2, label='y=x')
    ax.set_xlabel('|Correlation| without Z', fontsize=12, fontweight='bold')
    ax.set_ylabel('|Correlation| with Z (residual)', fontsize=12, fontweight='bold')
    ax.set_title('Pairwise Cause Correlations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Z variance
    ax = axes[1]
    ax.bar(range(len(Z_var)), Z_var, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='Collapse threshold')
    ax.axhline(y=0.95, color='orange', linestyle='--', linewidth=2, label='Prior level')
    ax.set_xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance', fontsize=12, fontweight='bold')
    ax.set_title('Z Variance per Dimension', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('test4_confounder_validity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'unconditional_corrs': unconditional_corrs,
        'conditional_corrs': conditional_corrs,
        'z_variance': Z_var
    }


# ============================================================
# Main: Run All PPC Tests
# ============================================================

def run_all_ppc_tests(
    model_obj,
    A_train, X_train,
    A_held, X_held
):
    """
    运行所有PPC测试
    
    Args:
        model_obj: 训练好的PyroDVAE
        A_train: [n_users, T_train, n_categories] (Day 1-7)
        X_train: [n_users, T_train, x_dim]
        A_held: [n_users, T_held, n_categories] (Day 8-9)
        X_held: [n_users, T_held, x_dim]
    
    Returns:
        all_results: dict containing all test results
    """
    print("\n" + "="*70)
    print(" POSTERIOR PREDICTIVE CHECKING FOR TIME-SERIES DECONFOUNDER")
    print("="*70)
    print(f"\nTraining data: {A_train.shape[0]} users × {A_train.shape[1]} days × {A_train.shape[2]} categories")
    print(f"Held-out data: {A_held.shape[0]} users × {A_held.shape[1]} days × {A_held.shape[2]} categories")
    
    all_results = {}
    
    # Test 1: Held-out likelihood
    print("\n" + "🔬"*35)
    test1_results = test1_heldout_predictive_likelihood(
        model_obj, A_train, X_train, A_held, X_held
    )
    all_results['test1'] = test1_results
    
    # Test 2: LOCO
    print("\n" + "🔬"*35)
    test2_results = test2_leave_one_cause_out(
        model_obj, A_train, X_train, num_test_causes=20
    )
    all_results['test2'] = test2_results
    
    # Test 3: Temporal forecasting
    print("\n" + "🔬"*35)
    test3_results = test3_temporal_forecasting(
        model_obj, A_train, X_train, A_held, X_held, forecast_steps=[1, 2]
    )
    all_results['test3'] = test3_results
    
    # Test 4: Validity
    print("\n" + "🔬"*35)
    test4_results = test4_substitute_confounder_validity(
        model_obj, A_train, X_train, A_held, X_held
    )
    all_results['test4'] = test4_results
    
    # -------- Final Summary --------
    print("\n" + "="*70)
    print(" FINAL PPC SUMMARY")
    print("="*70)
    
    print("\n📊 Test 1 - Held-out Likelihood:")
    improvement = test1_results['control_z'] - test1_results['no_control']
    if improvement > 0:
        print(f"   ✅ Improvement: {improvement:.2f} (Z helps!)")
    else:
        print(f"   ⚠️  No improvement: {improvement:.2f}")
    
    print("\n📊 Test 2 - LOCO:")
    if len(test2_results) > 0:
        avg_imp = (test2_results['ll_with_z'] - test2_results['ll_without_z']).mean()
        print(f"   Average improvement: {avg_imp:.3f}")
        if avg_imp > 0:
            print("   ✅ Z provides information beyond naive baseline")
        else:
            print("   ⚠️  Z not helpful")
    
    print("\n📊 Test 3 - Temporal Forecasting:")
    if len(test3_results) > 0:
        best_dynamic = test3_results['ll_dynamic'].max()
        best_baseline = test3_results['ll_static'].max()
        print(f"   Dynamic: {best_dynamic:.3f}")
        print(f"   Baseline: {best_baseline:.3f}")
        if best_dynamic > best_baseline:
            print("   ✅ Dynamic model outperforms static")
        else:
            print("   ⚠️  No temporal advantage")
    
    print("\n📊 Test 4 - Substitute Validity:")
    z_var_mean = test4_results['z_variance'].mean()
    corr_reduction = (
        np.mean(test4_results['unconditional_corrs']) - 
        np.mean(test4_results['conditional_corrs'])
    )
    print(f"   Z variance: {z_var_mean:.4f}")
    print(f"   Correlation reduction: {corr_reduction:.3f}")
    
    if z_var_mean > 0.1 and corr_reduction > 0:
        print("   ✅ Z is a valid substitute confounder")
    else:
        print("   ⚠️  Z may not be sufficient")
    
    print("\n" + "="*70)
    print(" RECOMMENDATION")
    print("="*70)
    
    # 判断模型质量
    score = 0
    if improvement > 0: score += 1
    if len(test2_results) > 0 and avg_imp > 0: score += 1
    if len(test3_results) > 0 and best_dynamic > best_baseline: score += 1
    if z_var_mean > 0.1 and corr_reduction > 0: score += 1
    
    if score >= 3:
        print("\n✅ PASS - 你的Dynamic Deconfounder quality高！")
        print("   可以进行causal inference了")
    elif score >= 2:
        print("\n⚠️  MARGINAL - 模型有一定效果，但有改进空间")
        print("   建议：")
        print("   - 增加latent_dim")
        print("   - 调整KL annealing")
        print("   - 检查architecture")
    else:
        print("\n❌ FAIL - Z不是good substitute")
        print("   建议：")
        print("   - 尝试不同的factor model")
        print("   - 增加model capacity")
        print("   - 检查数据是否有足够信号")
    
    return all_results


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    """
    Example usage
    """
    from model import PyroDVAE
    
    # 假设你已经训练好model
    model = PyroDVAE(input_dim=100, latent_dim=10, hidden_dim=64, x_dim=5)
    # model = torch.load('trained_model.pt')
    
    # Load data
    # A_train: [n_users, 7, n_categories]  # Day 1-7
    # X_train: [n_users, 7, x_dim]
    # A_held: [n_users, 2, n_categories]   # Day 8-9
    # X_held: [n_users, 2, x_dim]
    
    # Run all tests
    # all_results = run_all_ppc_tests(
    #     model_obj=model,
    #     A_train=A_train,
    #     X_train=X_train,
    #     A_held=A_held,
    #     X_held=X_held
    # )

