"""多変量の walk-forward 予測（パージ + エンバーゴ付き）とグループ単位の置換重要度。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class PurgedRidge:
    """学習期間の末尾 val_frac を時系列の検証区間にして alpha を選ぶ Ridge。

    RidgeCV の LOO は重複ターゲット（多日リターン）では隣接行からリークし、alpha が小さく選ばれて過学習する。
    """

    def __init__(self, h: int = 21, alphas=np.logspace(0, 6, 25), val_frac: float = 0.25):
        self.h, self.alphas, self.val_frac = h, alphas, val_frac

    def fit(self, X, y):
        n = len(X)
        v0 = int(n * (1 - self.val_frac))
        Xa, ya, Xv, yv = X.iloc[:v0 - self.h], y.iloc[:v0 - self.h], X.iloc[v0:], y.iloc[v0:]
        best, best_err = self.alphas[-1], np.inf
        for a in self.alphas:
            m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xa, ya)
            err = float(((m.predict(Xv) - yv) ** 2).mean())
            if err < best_err:
                best, best_err = a, err
        self.alpha_ = best
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=best)).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


MODELS = {
    "ridge": lambda h: PurgedRidge(h=h),
    "hgb": lambda h: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300,
                                                 min_samples_leaf=200, l2_regularization=1.0, random_state=0),
}


def usable_matrix(X: pd.DataFrame, y: pd.Series, max_nan_frac: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    """欠損の多い特徴量を落とし、残りが揃う行だけを使う。"""
    valid_y = y.notna()
    nan_frac = X[valid_y].isna().mean()
    cols = nan_frac[nan_frac <= max_nan_frac].index
    d = pd.concat([X[cols], y.rename("__y")], axis=1).dropna()
    return d[cols], d["__y"]


def _spearman(a, b) -> float:
    a, b = pd.Series(np.asarray(a)), pd.Series(np.asarray(b))  # index を揃えず位置で対応させる
    return float(a.rank().corr(b.rank())) if a.nunique() > 1 else np.nan


def walk_forward(X: pd.DataFrame, y: pd.Series, h: int, model: str, groups: dict[str, list[str]],
                 min_train: int = 1260, test_size: int = 252, n_perm: int = 5, seed: int = 0) -> dict:
    """拡張窓で学習し次の test_size 日を予測。学習末尾は h 日パージ（ターゲットの重複を除去）。"""
    Xu, yu = usable_matrix(X, y)
    rng = np.random.default_rng(seed)
    preds, folds, imps = [], [], []
    n, cal = len(Xu), X.index
    for start in range(min_train, n, test_size):
        # 行 t のターゲットは t+h まで使うので、テスト開始の h 営業日前で学習を打ち切る（暦ベースでパージ）
        cutoff = cal[cal.get_loc(Xu.index[start]) - h]
        train = Xu.index < cutoff
        if train.sum() < min_train // 2:
            continue
        Xtr, ytr = Xu[train], yu[train]
        Xte, yte = Xu.iloc[start:start + test_size], yu.iloc[start:start + test_size]
        lo, hi = ytr.quantile([0.01, 0.99])
        m = MODELS[model](h).fit(Xtr, ytr.clip(lo, hi))
        p = m.predict(Xte)
        preds.append(pd.Series(p, index=Xte.index))
        base_ic = _spearman(p, yte)
        folds.append(dict(test_start=Xte.index[0], test_end=Xte.index[-1], n_train=len(Xtr),
                          ic=base_ic, train_mean=float(ytr.mean())))
        # グループ置換重要度: 相関の強い特徴量群を同じ行置換でまとめて壊す
        row = {}
        for g, cols in groups.items():
            cols = [c for c in cols if c in Xte]
            if not cols:
                continue
            drops = []
            for _ in range(n_perm):
                Xp = Xte.copy()
                Xp[cols] = Xte[cols].to_numpy()[rng.permutation(len(Xte))]
                drops.append(base_ic - _spearman(m.predict(Xp), yte))
            row[g] = float(np.nanmean(drops))
        imps.append(row)

    pred = pd.concat(preds) if preds else pd.Series(dtype=float)
    yy = yu.reindex(pred.index)
    fold_df = pd.DataFrame(folds)
    # OOS R²: ゼロ予測基準と、学習期間平均（各フォールド）基準の両方
    mean_fc = pd.Series(np.nan, index=pred.index)
    for f in folds:
        mean_fc[(mean_fc.index >= f["test_start"]) & (mean_fc.index <= f["test_end"])] = f["train_mean"]
    sse = ((yy - pred) ** 2).sum()
    return dict(
        pred=pred, folds=fold_df, features=list(Xu.columns),
        importance=pd.DataFrame(imps).mean().sort_values(ascending=False) if imps else pd.Series(dtype=float),
        importance_pos_frac=(pd.DataFrame(imps) > 0).mean() if imps else pd.Series(dtype=float),
        oos_ic=_spearman(pred, yy),  # 全期間プール（フォールド間の予測水準差に影響される）
        mean_fold_ic=float(fold_df["ic"].mean()) if len(fold_df) else np.nan,
        oos_r2_zero=float(1 - sse / (yy ** 2).sum()),
        oos_r2_mean=float(1 - sse / ((yy - mean_fc) ** 2).sum()),
        hit_rate=float((np.sign(pred) == np.sign(yy)).mean()),
        frac_folds_ic_pos=float((fold_df["ic"] > 0).mean()) if len(fold_df) else np.nan,
    )
