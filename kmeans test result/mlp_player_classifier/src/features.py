import numpy as np
import pandas as pd

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 衍生 AFK_ratio & active_minutes
    df["AFK_ratio"] = df["afktime"] / 1800.0
    df["active_minutes"] = np.maximum(0.0, 30.0 - df["afktime"] / 60.0)
    return df

def filter_rows(df: pd.DataFrame, drop_afk_ratio_eq1: bool = True) -> pd.DataFrame:
    if drop_afk_ratio_eq1:
        df = df[df["AFK_ratio"] < 1.0].copy()
    return df

def winsorize(df: pd.DataFrame, cols, p_low=0.0, p_high=0.995):
    df = df.copy()
    for c in cols:
        lo = df[c].quantile(p_low)
        hi = df[c].quantile(p_high)
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df

def log1p_cols(df: pd.DataFrame, cols):
    df = df.copy()
    df[cols] = np.log1p(df[cols])
    return df

def robust_scale(df: pd.DataFrame, cols):
    df = df.copy()
    med = df[cols].median()
    iqr = (df[cols].quantile(0.75) - df[cols].quantile(0.25)).replace(0, 1.0)
    df[cols] = (df[cols] - med) / iqr
    return df
