import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

class TabularDataset:
    def __init__(self, df, feature_cols, label_col, class_weight="balanced"):
        self.X = df[feature_cols].to_numpy(np.float32)
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(df[label_col].astype(str))
        self.classes_ = list(self.le.classes_)
        if class_weight == "balanced":
            # 1/freq 權重
            _, counts = np.unique(self.y, return_counts=True)
            self.class_weights = (counts.sum() / (len(counts) * counts)).astype(np.float32)
        else:
            self.class_weights = np.ones(len(self.le.classes_), dtype=np.float32)

def group_split_by_player(df, group_col, test_players_ratio=0.2, seed=42):
    # 以玩家為單位切 Train/Test，避免同一玩家出現在兩邊
    gss = GroupShuffleSplit(n_splits=1, test_size=test_players_ratio, random_state=seed)
    groups = df[group_col].astype(str)
    idx_train, idx_test = next(gss.split(df, groups=groups))
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()
