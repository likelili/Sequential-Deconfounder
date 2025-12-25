import numpy as np
import pandas as pd
import torch


# ==== Data Processing ====
def process_alibaba_data(raw_data, keep_count=True):
    """
    Process Alibaba data into user-timestep-category format

    Input: DataFrame with columns [user, timestep, category_id, exposure_count]
    Output: Tensor A[user, timestep, category]
    """
    # Get dimensions
    users = raw_data['user'].unique()
    timesteps = raw_data['time_stamp'].unique()
    categories = raw_data['cate_id'].unique()

    n_users = len(users)
    n_timesteps = len(timesteps)
    n_categories = len(categories)

    print(f"Data shape: {n_users} users x {n_timesteps} timesteps x {n_categories} categories")

    # Create mapping dictionaries
    user_map = {u: i for i, u in enumerate(users)}
    time_map = {t: i for i, t in enumerate(sorted(timesteps))}
    cat_map = {c: i for i, c in enumerate(categories)}

    # Initialize tensor
    A = np.zeros((n_users, n_timesteps, n_categories))

    # Fill tensor
    for _, row in raw_data.iterrows():
        u_idx = user_map[row['user']]
        t_idx = time_map[row['time_stamp']]
        c_idx = cat_map[row['cate_id']]
        A[u_idx, t_idx, c_idx] = row['exposure_count']

    if keep_count==False:
      A = (A > 0).astype(int)

    return A, users, timesteps, categories


def behaviors_to_X_window(
    behaviors,
    behav_cols,
    users_all,
    user2idx,
    exposure_start,
    exposure_end,
):
    """
    behaviors: behavioral log
    behav_cols: ['behav__pv','behav__cart','behav__fav', ...]
    exposure_start / exposure_end: datetime.date
    Output:
      X_window: torch.FloatTensor, shape (U, T_window, D_x)
      lag_cols: list[str]
      dates_window: list[date]
    """

    # -------- 0. Preprocess time --------
    df = behaviors.copy()
    df["date"] = pd.to_datetime(
        df["time_stamp"], unit="s", errors="coerce"
    ).dt.date

    # -------- 1. Day-level aggregation (over the full timeline) --------
    daily = (
        df[df[behav_cols].sum(axis=1) > 0]
        .groupby(["user", "date"])
        .agg(
            pv_count=("behav__pv", "sum"),
            cart_count=("behav__cart", "sum"),
            fav_count=("behav__fav", "sum"),
            distinct_cate=("cate", "nunique"),
        )
        .reset_index()
        .sort_values(["user", "date"])
    )

    base_cols = ["pv_count", "cart_count", "fav_count", "distinct_cate"]
    lag_cols  = [f"{c}_lag1" for c in base_cols]

    # -------- 2. lag(1) on the full timeline --------
    for c in base_cols:
        daily[f"{c}_lag1"] = daily.groupby("user")[c].shift(1)

    # log1p (keep NaN as is)
    for c in lag_cols:
        daily[c] = np.log1p(daily[c])

    # -------- 3. Construct the window timeline --------
    dates_window = list(
        pd.date_range(exposure_start, exposure_end, freq="D").date
    )
    date2idx = {d: i for i, d in enumerate(dates_window)}

    U = len(users_all)
    T = len(dates_window)
    D_x = len(lag_cols)

    # -------- 4. Initialize tensor --------
    X = np.zeros((U, T, D_x), dtype=np.float32)

    # -------- 5. Select data within the window and map indices --------
    daily_w = daily[
        (daily["date"] >= exposure_start) &
        (daily["date"] <= exposure_end)
    ].copy()

    daily_w["ui"] = daily_w["user"].map(user2idx)
    daily_w["ti"] = daily_w["date"].map(date2idx)

    daily_w = daily_w.dropna(subset=["ui", "ti"]).copy()
    daily_w["ui"] = daily_w["ui"].astype(int)
    daily_w["ti"] = daily_w["ti"].astype(int)

    # -------- 6. Write into tensor (NaN → 0) --------
    for j, c in enumerate(lag_cols):
        vals = daily_w[c].fillna(0.0).values.astype(np.float32)
        X[
            daily_w["ui"].values,
            daily_w["ti"].values,
            j
        ] = vals

    print(X.shape)
    return torch.from_numpy(X), lag_cols, dates_window



def exposure_table_to_tensor(exposure, users_all, user2idx, cate2idx, dates_all, date2idx):
    """
    exposure: full exposure DataFrame
              columns = ['user', 'time_stamp', 'cate_id', 'exposure_count']
    Output:
      A_all: torch.FloatTensor, shape (U, T, D)
    """

    U = len(users_all)
    T = len(dates_all)
    D = len(cate2idx)

    # Initialize: no record = no exposure = 0
    A = np.zeros((U, T, D), dtype=np.float32)

    df = exposure.copy()

    # Map to indices (unmapped values will be NaN)
    df["ui"] = df["user"].map(user2idx)
    df["ti"] = df["date"].map(date2idx)
    df["ci"] = df["cate_id"].map(cate2idx)

    # Drop invalid rows (unknown user / cate / date)
    df = df.dropna(subset=["ui", "ti", "ci"]).copy()
    df["ui"] = df["ui"].astype(int)
    df["ti"] = df["ti"].astype(int)
    df["ci"] = df["ci"].astype(int)

    print("ui NaN rate:", df["ui"].isna().mean())
    print("ti NaN rate:", df["ti"].isna().mean())
    print("ci NaN rate:", df["ci"].isna().mean())
    # If multiple records exist for the same (user, date, cate), aggregate first
    agg = (
        df.groupby(["ui", "ti", "ci"], as_index=False)["exposure_count"]
        .sum()
    )

    # Write into tensor
    A[
        agg["ui"].values,
        agg["ti"].values,
        agg["ci"].values
    ] = agg["exposure_count"].values.astype(np.float32)

    return torch.from_numpy(A)



def user_process(A_tensor, X_tensor, users_all):
    """
    A_tensor: torch.Tensor or np.ndarray, shape (U, T, D_A)
    X_tensor: torch.Tensor or np.ndarray, shape (U, T, D_x)
    users_all: list of user ids, length U, fixed order
    """

    # -------- 1. one-hot encode user static features --------
    user_feat_oh = pd.get_dummies(
        user_static,
        columns=user_static_cols,
        dummy_na=False
    ).set_index("userid")

    # -------- 2. align to panel's user axis (critical) --------
    user_feat_oh = user_feat_oh.reindex(users_all)

    if user_feat_oh.isna().any().any():
        print("[Warning] NaN found in user static features after reindex")
        user_feat_oh = user_feat_oh.fillna(0.0)

    # -------- 3. sanity check --------
    U, T, _ = X_tensor.shape
    assert user_feat_oh.shape[0] == U

    # -------- 4. expand to (U, T, D_user) --------
    user_feat_np = user_feat_oh.values.astype(np.float32)
    D_user = user_feat_np.shape[1]

    user_feat_expanded = np.repeat(
        user_feat_np[:, None, :],
        repeats=T,
        axis=1
    )

    # -------- 5. concatenate --------
    if hasattr(X_tensor, "cpu"):  # torch tensor
        X_np = X_tensor.cpu().numpy()
    else:
        X_np = X_tensor

    X_np = np.concatenate(
        [X_np, user_feat_expanded],
        axis=2
    )

    # -------- 6. final checks --------
    assert A_tensor.shape[0] == X_np.shape[0]
    assert A_tensor.shape[1] == X_np.shape[1]
    assert np.isfinite(X_np).all()

    return A_tensor, X_np



def save_input(A_tensor, X_tensor, text):
  if "torch" in str(type(A_tensor)):
      A_np = A_tensor.cpu().numpy()
  else:
      A_np = A_tensor

  if "torch" in str(type(X_tensor)):
      X_np = X_tensor.cpu().numpy()
  else:
      X_np = X_tensor

  np.savez(
      data_path+f"/dvae_inputs_{text}.npz",
      A=A_np,
      X=X_np,
      users=np.array(users_all),
      dates=np.array(dates_all),
  )