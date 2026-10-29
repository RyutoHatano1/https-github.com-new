"""図と Markdown レポートの出力。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def md_table(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = False) -> str:
    """tabulate に依存しない簡易 Markdown 表。"""
    d = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d")
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def plot_ic_heatmap(ic: pd.DataFrame, path: Path, top: int = 35):
    piv = ic.pivot_table(index="feature", columns="target", values="ic_disc")
    rob = ic.pivot_table(index="feature", columns="target", values="robust", aggfunc="max").reindex_like(piv)
    order = ic.groupby("feature")["p_hac"].min().sort_values().index[:top]
    piv, rob = piv.loc[order], rob.loc[order]
    fig, ax = plt.subplots(figsize=(1.4 * piv.shape[1] + 4, 0.32 * len(piv) + 1.5))
    lim = np.nanmax(np.abs(piv.to_numpy())) or 0.1
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}{'*' if rob.iat[i, j] == 1 else ''}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman IC (discovery)")
    ax.set_title("IC (discovery); * = FDR<5% & holdout replicates")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rolling_betas(b: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(len(b.columns), 1, figsize=(10, 2.2 * len(b.columns)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), b.columns):
        ax.plot(b.index, b[c], lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"beta {c}")
    fig.suptitle("SOX daily return: rolling 252d univariate betas")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_leverage_drag(ld: dict, path: Path):
    b = ld["blocks"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(b["var"], b["gap"], s=12, alpha=0.7)
    x = np.linspace(0, b["var"].max(), 50)
    ax.plot(x, ld["theory_slope"] * x, "k--", label=f"theory slope {ld['theory_slope']:.1f}")
    ax.plot(x, ld["intercept_ann"] * 63 / 252 + ld["slope"] * x, "r-", label=f"fit slope {ld['slope']:.2f}")
    ax.set_xlabel("sum of SOX daily r^2 in 63d window")
    ax.set_ylabel("log(SOXL) - 3*log(SOX) over window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity(runs: dict[str, pd.DataFrame], names: list[str], split: pd.Timestamp, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for n in names:
        eq = (1 + runs[n]["ret"]).cumprod()
        ax.plot(eq.index, eq, lw=1, label=n)
    ax.set_yscale("log")
    ax.axvline(split, color="k", ls="--", lw=0.8)
    ax.text(split, ax.get_ylim()[1], " holdout →", va="top", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("Equity (log scale), costs included")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(imp) + 1.5))
    imp.plot.barh(ax=ax)
    ax.axvline(0, color="k", lw=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("drop in OOS Spearman IC when group permuted")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
