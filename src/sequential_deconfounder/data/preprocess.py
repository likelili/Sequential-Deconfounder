import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class RawPaths:
    samples_csv: Path
    users_csv: Path
    features_csv: Path
    behaviors_csv: Path


def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        sample = series.dropna()
        if sample.empty:
            return pd.to_datetime(series, errors="coerce")
        unit = "ms" if sample.iloc[0] > 1e12 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _to_date(series: pd.Series) -> pd.Series:
    return _to_datetime(series).dt.date


def load_raw_tables(paths: RawPaths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(paths.samples_csv)
    user = pd.read_csv(paths.users_csv)
    feature = pd.read_csv(paths.features_csv)
    behaviors = pd.read_csv(paths.behaviors_csv)
    return sample, user, feature, behaviors


def normalize_ids(feature: pd.DataFrame, behaviors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature = feature.copy()
    behaviors = behaviors.copy()

    def convert_float_id(series: pd.Series) -> pd.Series:
        s = series.copy()
        fractional = (s.dropna() % 1)
        if (fractional != 0).any():
            print("Warning: ID values have non-zero decimals; casting to int strings.")
        return s.apply(lambda x: str(int(x)) if pd.notna(x) else x)

    feature["brand"] = convert_float_id(feature["brand"])
    behaviors["brand"] = behaviors["brand"].astype("string")
    return feature, behaviors


def merge_sample_features(sample: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(sample, feature[["adgroup_id", "cate_id", "brand"]], on="adgroup_id", how="left")


def build_outcomes(sample: pd.DataFrame, behaviors: pd.DataFrame) -> pd.DataFrame:
    brand_pairs = sample[["user", "brand"]].drop_duplicates()
    cate_pairs = sample[["user", "cate_id"]].drop_duplicates()
    cate_pairs = cate_pairs.rename(columns={"cate_id": "cate"})

    brand_pairs["brand"] = brand_pairs["brand"].apply(lambda x: str(float(x)) if pd.notna(x) else x)
    cate_pairs["cate"] = cate_pairs["cate"].apply(lambda x: str(float(x)) if pd.notna(x) else x)

    behaviors = behaviors.copy()
    behaviors["cate"] = behaviors["cate"].apply(lambda x: str(float(x)) if pd.notna(x) else x)
    behaviors["brand"] = behaviors["brand"].apply(lambda x: str(float(x)) if pd.notna(x) else x)

    brand_matched = behaviors.merge(brand_pairs, on=["user", "brand"], how="inner")
    brand_matched["match_type"] = "brand_seen"
    cate_matched = behaviors.merge(cate_pairs, on=["user", "cate"], how="inner")
    cate_matched["match_type"] = "category_seen"

    filtered = pd.concat([brand_matched, cate_matched], ignore_index=True)
    filtered["count"] = filtered.groupby(filtered.columns.tolist()).transform("size")
    filtered["match_type"] = filtered.apply(
        lambda r: "both" if r["count"] > 1 else r["match_type"],
        axis=1,
    )
    filtered = filtered.drop(columns=["count"]).drop_duplicates()
    filtered = pd.get_dummies(filtered, columns=["match_type"], prefix="")
    return filtered


def build_exposure_table(sample: pd.DataFrame) -> pd.DataFrame:
    sample_cate = sample.copy()
    sample_cate["time_stamp"] = _to_date(sample_cate["time_stamp"])
    exposure = (
        sample_cate
        .groupby(["user", "time_stamp", "cate_id"])
        .size()
        .reset_index(name="exposure_count")
        .sort_values(by=["user", "time_stamp", "cate_id"], ascending=True)
        .reset_index(drop=True)
    )
    return exposure


def _ensure_behavior_dummies(behaviors: pd.DataFrame) -> pd.DataFrame:
    if "btag" in behaviors.columns:
        return pd.get_dummies(behaviors, columns=["btag"], prefix="behav_")
    return behaviors


def build_behavior_tables(behaviors: pd.DataFrame) -> dict:
    behaviors = behaviors.copy()
    behaviors = _ensure_behavior_dummies(behaviors)
    behaviors["date"] = _to_date(behaviors["time_stamp"])

    tables = {}
    for tag in ["buy", "cart", "fav", "pv"]:
        col = f"behav__{tag}"
        if col not in behaviors.columns:
            continue
        table = (
            behaviors[behaviors[col] == True]  # noqa: E712
            .groupby(["user", "date", "cate"])
            .size()
            .reset_index(name="exposure_count")
            .sort_values(by=["user", "date", "cate"], ascending=True)
            .reset_index(drop=True)
        )
        tables[tag] = table
    return tables


def build_daily_engagement(behaviors: pd.DataFrame) -> pd.DataFrame:
    behaviors = _ensure_behavior_dummies(behaviors).copy()
    behaviors["date"] = _to_date(behaviors["time_stamp"])

    behav_cols = [c for c in behaviors.columns if c.startswith("behav__")]
    behav_engage = behaviors[behaviors[behav_cols].sum(axis=1) > 0].copy()

    daily = (
        behav_engage
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
    for c in base_cols:
        daily[f"{c}_lag1"] = daily.groupby("user")[c].shift(1)

    lag_cols = [f"{c}_lag1" for c in base_cols]
    daily[lag_cols] = daily[lag_cols].fillna(0.0)
    return daily


def build_panel_tensors(
    exposure: pd.DataFrame,
    daily_engage: pd.DataFrame,
    user: pd.DataFrame,
    exposure_start: str,
    exposure_end: str,
    max_users: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List, List]:
    exposure = exposure.copy()
    exposure["date"] = _to_date(exposure["time_stamp"])

    exposure_start = pd.to_datetime(exposure_start).date()
    exposure_end = pd.to_datetime(exposure_end).date()

    exp_window = exposure[
        (exposure["date"] >= exposure_start) &
        (exposure["date"] < exposure_end)
    ].copy()

    if max_users is not None:
        rng = np.random.default_rng(seed)
        user_ids = exp_window["user"].unique()
        keep_users = rng.choice(user_ids, max_users, replace=False)
        exp_window = exp_window[exp_window["user"].isin(keep_users)]

    cate_list = sorted(exp_window["cate_id"].unique())
    cate2idx = {c: i for i, c in enumerate(cate_list)}

    A_df = (
        exp_window.assign(cate_idx=lambda x: x["cate_id"].map(cate2idx))
        .pivot_table(
            index=["user", "date"],
            columns="cate_idx",
            values="exposure_count",
            fill_value=0,
        )
        .reset_index()
    )

    lag_cols = [c for c in daily_engage.columns if c.endswith("_lag1")]
    X_df = daily_engage[["user", "date"] + lag_cols].copy()
    X_df = X_df[
        (X_df["date"] >= exposure_start) &
        (X_df["date"] < exposure_end)
    ].copy()

    keys = A_df[["user", "date"]].drop_duplicates()
    full_df = (
        keys
        .merge(A_df, on=["user", "date"], how="left")
        .merge(X_df, on=["user", "date"], how="left")
        .fillna(0)
    )

    cate_cols = sorted([c for c in full_df.columns if isinstance(c, int)])
    users = sorted(full_df["user"].unique())
    dates = sorted(full_df["date"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    date2idx = {d: i for i, d in enumerate(dates)}

    U, T, K = len(users), len(dates), len(cate_cols)
    D = len(lag_cols)

    A_tensor = np.zeros((U, T, K), dtype=np.float32)
    X_tensor = np.zeros((U, T, D), dtype=np.float32)

    for _, row in full_df.iterrows():
        u = user2idx[row["user"]]
        t = date2idx[row["date"]]
        A_tensor[u, t, :] = row[cate_cols].values
        X_tensor[u, t, :] = row[lag_cols].values

    user_static_cols = [
        "final_gender_code",
        "age_level",
        "shopping_level",
        "occupation",
        "cms_segid",
    ]
    user_static = user[["userid"] + user_static_cols].copy()
    user_feat_oh = pd.get_dummies(
        user_static,
        columns=user_static_cols,
        dummy_na=False,
    ).set_index("userid")
    user_feat_oh = user_feat_oh.reindex(users).fillna(0.0)
    user_feat_np = user_feat_oh.values.astype(np.float32)
    user_feat_expanded = np.repeat(user_feat_np[:, None, :], repeats=T, axis=1)
    X_tensor = np.concatenate([X_tensor, user_feat_expanded], axis=2)

    return A_tensor, X_tensor, users, dates


def save_npz(out_path: Path, A: np.ndarray, X: np.ndarray, users: Iterable, dates: Iterable) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        A=A,
        X=X,
        users=np.array(list(users)),
        dates=np.array(list(dates)),
    )


def _load_yaml_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required for --config. Install with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    cfg = _load_yaml_config(Path(args.config))
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def run_pipeline(args: argparse.Namespace) -> None:
    args = _apply_config_overrides(args)
    required = ["samples_csv", "users_csv", "features_csv", "behaviors_csv", "out_dir"]
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    paths = RawPaths(
        samples_csv=Path(args.samples_csv),
        users_csv=Path(args.users_csv),
        features_csv=Path(args.features_csv),
        behaviors_csv=Path(args.behaviors_csv),
    )
    sample, user, feature, behaviors = load_raw_tables(paths)

    behav_dt = _to_datetime(behaviors["time_stamp"])
    behaviors = behaviors[behav_dt.dt.year == 2017]
    feature, behaviors = normalize_ids(feature, behaviors)
    sample = merge_sample_features(sample, feature)

    outcomes = build_outcomes(sample, behaviors)
    exposure = build_exposure_table(sample)
    behavior_tables = build_behavior_tables(behaviors)
    daily_engage = build_daily_engagement(behaviors)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(out_dir / "outcomes.csv", index=False)
    exposure.to_csv(out_dir / "exposure.csv", index=False)
    for name, table in behavior_tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)

    A_tensor, X_tensor, users, dates = build_panel_tensors(
        exposure=exposure,
        daily_engage=daily_engage,
        user=user,
        exposure_start=args.exposure_start,
        exposure_end=args.exposure_end,
        max_users=args.max_users,
        seed=args.seed,
    )
    save_npz(out_dir / "dvae_inputs.npz", A_tensor, X_tensor, users, dates)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Alibaba display ad data.")
    parser.add_argument("--config", help="Path to YAML config file", default=None)
    parser.add_argument("--samples_csv", help="Path to samples.csv")
    parser.add_argument("--users_csv", help="Path to users.csv")
    parser.add_argument("--features_csv", help="Path to features.csv")
    parser.add_argument("--behaviors_csv", help="Path to behavior_log.csv")
    parser.add_argument("--out_dir", help="Output directory for processed files")
    parser.add_argument("--exposure_start", default="2017-05-05", help="Start date (inclusive)")
    parser.add_argument("--exposure_end", default="2017-05-12", help="End date (exclusive)")
    parser.add_argument("--max_users", type=int, default=None, help="Optional subsample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    return parser


if __name__ == "__main__":
    run_pipeline(build_argparser().parse_args())
