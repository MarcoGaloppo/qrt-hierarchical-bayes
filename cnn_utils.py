"""Shared code for CNNApproach.ipynb.

Contents
--------
1. Date reconstruction: rebuilds the shuffled training dates into stretches of
   consecutive days (PIECE, POS). The logic is the same as in
   BaysianHierarchicalApproach.ipynb, so the folds match the HBM's.
2. Folds: GroupKFold over stretches, so no validation date has a neighbour in training.
3. Inputs: vol-scaled returns, arcsinh-scaled signed volumes, group and portfolio indices.
4. CNNs:four architectures of the notebook (vanilla, volume-aware, no-pool, multi-scale).
   Each can be conditioned on the portfolio's GROUP (and, optionally on the portfolio itself)
   through a FiLM layer.
5. Training and evaluation: one generic fold trainer, cross-validation (optionally
   with folds in parallel), summaries, loss plots and a paired comparison with the HBM.
"""
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RET = [f"RET_{i}" for i in range(1, 21)]             # RET_1 = most recent day
VOL = [f"SIGNED_VOLUME_{i}" for i in range(2, 21)]   # SIGNED_VOLUME_1 is mostly missing, left out
EPSILON = 1e-12

# HBM accuracy on the same five folds used for paired comparisons
HBM_FOLD_ACC = np.array([0.5186, 0.5203, 0.5249, 0.5258, 0.5271])


# =============================================================================
# 1. Date reconstruction
# =============================================================================
# The functions reproduce the companion notebook exactly

def next_day_votes(X, k=1, overlap=15, ratio=0.3):
    #For every portfolio, each row votes for the date whose window looks shifted by k days.
    votes = []
    for a, g in X.groupby("ALLOCATION"):
        R = np.nan_to_num(g[RET].to_numpy())
        original = R[:, 0:overlap] / (np.linalg.norm(R[:, 0:overlap]) + EPSILON)
        shifted = R[:, k:overlap + k] / (np.linalg.norm(R[:, k:overlap + k]) + EPSILON)
        d, j = cKDTree(shifted).query(original, k=2)
        clear = d[:, 0] < ratio * d[:, 1]
        votes.append(pd.DataFrame({"ALLOCATION": a, "TS": g["TS"].values[clear],
                                   "NEXT_TS": g["TS"].values[j[clear, 0]]}))
    return pd.concat(votes, ignore_index=True)


def pick_links(votes, min_votes=10):
    #Keep a date's top candidate if enough portfolios vote for it and it has a majority.
    n = votes.groupby(["TS", "NEXT_TS"]).size().rename("n").reset_index()
    n["share"] = n["n"] / n.groupby("TS")["n"].transform("sum")
    best = n.sort_values("n", ascending=False).drop_duplicates("TS")
    best = best[(best["n"] > min_votes) & (best["share"] > 0.5)]
    best.drop_duplicates("NEXT_TS", keep=False)          # not applied (see note above)
    return dict(zip(best["TS"], best["NEXT_TS"])), n


def verify_links(links, k, X, min_z=5.0, verbose=True):
    #Keep a link t -> u only if RET_1 on t reappears as RET_{k+1} on u: Spearman corr * sqrt(n) >= min_z
    by_date = {t: g.set_index("ALLOCATION") for t, g in X.groupby("TS")}
    good, zs = {}, {}
    for t, u in links.items():
        a = by_date[t]["RET_1"]
        b = by_date[u][f"RET_{k + 1}"].reindex(a.index)
        m = a.notna() & b.notna()
        n = m.sum()
        if n >= 10:
            z = a[m].corr(b[m], method="spearman") * np.sqrt(n)
            zs[t] = z
            if z >= min_z:
                good[t] = u
    if verbose:
        print(f"shift {k}: {len(links)} candidate links, {len(good)} pass (z >= {min_z})")
    return good, pd.Series(zs, dtype=float)


def build_chains(nxt, step, max_len=10_000):
    #Follow the links from every date without a predecessor. 
    chains = []
    for s in sorted(set(nxt) - set(nxt.values())):
        c, pos = [s], [0]
        while c[-1] in nxt:
            assert len(c) <= max_len, f"loop detected from {s}"
            pos.append(pos[-1] - step[c[-1]])
            c.append(nxt[c[-1]])
        chains.append((c, pos))
    chains.sort(key=lambda cp: (-len(cp[0]), cp[0][0]))
    return chains


def all_chains(nxt, step, all_dates):
    # Chains plus every unplaced date as a one-date chain.
    chains = build_chains(nxt, step)
    placed = {d for c, _ in chains for d in c}
    chains += [([d], [0]) for d in sorted(all_dates) if d not in placed]
    return chains


def unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + EPSILON)


def bridge_votes(X, k, from_dates, to_dates, overlap=10, max_dist=0.25):
    # Votes for k-day links, only from stretch ends to stretch starts
    From = X[X["TS"].isin(from_dates)]
    To = dict(tuple(X[X["TS"].isin(to_dates)].groupby("ALLOCATION")))
    votes = []
    for a, gf in From.groupby("ALLOCATION"):
        if a not in To:
            continue
        gt = To[a]
        this = unit(np.nan_to_num(gf[RET].to_numpy()[:, 0:overlap]))
        shifted = unit(np.nan_to_num(gt[RET].to_numpy()[:, k:overlap + k]))
        d, j = cKDTree(shifted).query(this, k=1)
        ok = d < max_dist
        votes.append(pd.DataFrame({"TS": gf["TS"].values[ok], "NEXT_TS": gt["TS"].values[j[ok]]}))
    return pd.concat(votes, ignore_index=True)


def _bridge_pass(X, nxt, step, chains, all_dates, shifts=range(2, 6), min_z=5.0, verbose=True):
    for k in shifts:
        end_of = {c[-1]: i for i, (c, _) in enumerate(chains)}
        start_of = {c[0]: i for i, (c, _) in enumerate(chains)}
        v = bridge_votes(X, k, set(end_of), set(start_of))
        v = v[v["TS"].map(end_of) != v["NEXT_TS"].map(start_of)]        # no self-loops
        cand, _ = pick_links(v, min_votes=30)
        new, _ = verify_links(cand, k, X, min_z=min_z, verbose=verbose)
        for t, u in new.items():
            nxt[t] = u
            step[t] = k
        chains = all_chains(nxt, step, all_dates)
        if verbose:
            print(f"  after k={k}: {len(new)} bridges, {len(chains)} stretches")
    return chains


def reconstruct_dates(X, overlap=15, ratio=0.3, min_z=5.0, verbose=True):
    #Return a copy of X with PIECE (stretch id, 0 = longest) and POS (day in the stretch, forward in time).
    X = X.copy()
    all_dates = X["TS"].unique()

    votes = next_day_votes(X, k=1, overlap=overlap, ratio=ratio)          # 1-day links
    links, _ = pick_links(votes)
    links_ok, _ = verify_links(links, k=1, X=X, min_z=min_z, verbose=verbose)
    nxt, step = dict(links_ok), {t: 1 for t in links_ok}
    chains = all_chains(nxt, step, all_dates)

    chains = _bridge_pass(X, nxt, step, chains, all_dates, min_z=min_z, verbose=verbose)  # 2-5 day bridges

    end_of = {c[-1]: i for i, (c, _) in enumerate(chains)}                 # rescue 1-day links
    start_of = {c[0]: i for i, (c, _) in enumerate(chains)}
    v1 = bridge_votes(X, 1, set(end_of), set(start_of), overlap=15, max_dist=0.25)
    v1 = v1[v1["TS"].map(end_of) != v1["NEXT_TS"].map(start_of)]
    cand1, _ = pick_links(v1, min_votes=30)
    new1, _ = verify_links(cand1, 1, X, min_z=min_z, verbose=verbose)
    for t, u in new1.items():
        nxt[t] = u
        step[t] = 1
    chains = all_chains(nxt, step, all_dates)
    chains = _bridge_pass(X, nxt, step, chains, all_dates, min_z=min_z, verbose=verbose)

    chains.sort(key=lambda cp: (-len(cp[0]), cp[0][0]))
    order = {d: (p, pos) for p, (c, ps) in enumerate(chains) for d, pos in zip(c, ps)}
    X["PIECE"] = X["TS"].map(lambda d: order[d][0])
    X["POS"] = X["TS"].map(lambda d: order[d][1])
    X["POS"] = X.groupby("PIECE")["POS"].transform("max") - X["POS"]     # make POS run forward in time

    if verbose:
        sizes = X.groupby("PIECE")["TS"].nunique().sort_values(ascending=False)
        print(f"{len(sizes)} stretches | longest 10: {sizes.head(10).tolist()} | "
              f">= 20 dates: {(sizes >= 20).sum()} | lone dates: {(sizes == 1).sum()}")
    assert X.groupby("TS")["PIECE"].nunique().max() == 1
    assert X.groupby(["PIECE", "POS"])["TS"].nunique().max() == 1
    return X


# =============================================================================
# 2. Folds
# =============================================================================

def make_folds(X, n_splits=5, verbose=True):
    #GroupKFold over stretches, with a check that no stretch appears on both sides
    folds = list(GroupKFold(n_splits=n_splits).split(X, groups=X["PIECE"]))
    for k, (tr, va) in enumerate(folds):
        shared = set(X["PIECE"].iloc[tr]) & set(X["PIECE"].iloc[va])
        assert not shared, f"fold {k} shares stretches: {shared}"
        if verbose:
            print(f"Fold {k}: train = {len(tr):,}, validation = {len(va):,}, "
                  f"stretches = {X['PIECE'].iloc[va].nunique()}")
    return folds


# =============================================================================
# 3. Inputs
# =============================================================================

def vol_stats(df):
    # Robust centre and scale of each signed-volume column (training rows only)
    out = {}
    for col in VOL:
        q25, q50, q75 = df[col].quantile([0.25, 0.5, 0.75])
        out[col] = (q50, (q75 - q25) / 1.349)
    return out


def prepare_inputs(df, vol_floor, vstats, portfolio_index):
    #Model inputs for a set of rows.

    #R: returns divided by the row's own 20-day vol (floored), missing go to 0.
    #V: arcsinh of the robust z-score of each signed volume, missing g oto 0.
    #g: group index 0..3.   p: portfolio index.   y: next-day return (if available).
    
    s = np.maximum(np.std(df[RET].to_numpy(), axis=1, ddof=1), vol_floor)
    R = np.nan_to_num(df[RET].to_numpy() / s[:, None], nan=0.0).astype(np.float32)
    V = np.zeros((len(df), len(VOL)), dtype=np.float32)
    for j, col in enumerate(VOL):
        centre, scale = vstats[col]
        V[:, j] = np.arcsinh((df[col] - centre) / scale).fillna(0.0).to_numpy()
    g = (df["GROUP"].astype(int).to_numpy() - 1).astype(np.int64)
    p = df["ALLOCATION"].map(portfolio_index).fillna(-1).astype(np.int64).to_numpy()
    y = df["target"].to_numpy() if "target" in df else None
    return {"R": R, "V": V, "g": g, "p": p, "y": y}


def build_cnn_folds(X, folds, verbose=True):
    #Per fold: training and validation inputs, with every scaling constant fitted on the training part only.
    portfolio_index = {a: i for i, a in enumerate(sorted(X["ALLOCATION"].unique()))}
    cnn_folds = []
    for k, (tr, va) in enumerate(folds):
        df_tr, df_va = X.iloc[tr], X.iloc[va]
        vol_floor = df_tr[RET].std(axis=1, ddof=1).quantile(0.01)
        vstats = vol_stats(df_tr)
        cnn_folds.append({"train": prepare_inputs(df_tr, vol_floor, vstats, portfolio_index),
                          "val": prepare_inputs(df_va, vol_floor, vstats, portfolio_index),
                          "val_idx": va, "vol_floor": vol_floor, "vstats": vstats,
                          "n_groups": int(X["GROUP"].nunique()), "n_portfolios": len(portfolio_index)})
        if verbose:
            print(f"Fold {k}: train {cnn_folds[-1]['train']['R'].shape}, val {cnn_folds[-1]['val']['R'].shape}, "
                  f"vol_floor = {vol_floor:.6g}")
    return cnn_folds


# =============================================================================
# 4. CNNs
# =============================================================================

class GroupFiLM(nn.Module):
    #Feature-wise linear modulation by group and by portfolio).

    #h -> h * (1 + scale) + shift, where scale and shift are learned per group. With
    #portfolios, each portfolio adds its own deviation on top of its group's values,
    #which mirrors the HBM's portfolio -> group pooling. All embeddings start at zero,
    #so an untrained model is exactly the corresponding model without groups. Weight
    #decay on the portfolio deviations plays the role of the HBM's shrinkage.
    #Works on (B, C) features or (B, C, L) feature maps.

    def __init__(self, n_features, n_groups=4, n_portfolios=278):
        super().__init__()
        self.group = nn.Embedding(n_groups, 2 * n_features)
        nn.init.zeros_(self.group.weight)
        self.portfolio = None
        if n_portfolios:
            self.portfolio = nn.Embedding(n_portfolios, 2 * n_features)
            nn.init.zeros_(self.portfolio.weight)

    def forward(self, h, g, p=None):
        film = self.group(g)
        if self.portfolio is not None and p is not None:
            film = film + self.portfolio(p.clamp(min=0)) * (p >= 0).unsqueeze(-1)   # unknown portfolio -> group only
        scale, shift = film.chunk(2, dim=-1)
        if h.dim() == 3:
            scale, shift = scale.unsqueeze(-1), shift.unsqueeze(-1)
        return h * (1 + scale) + shift


def _film(use_groups, n_features, n_groups, n_portfolios):
    return GroupFiLM(n_features, n_groups, n_portfolios) if use_groups else None


class ReturnCNN(nn.Module):
    # Vanilla CNN on the 20-day return history: conv(3) -> conv(3) -> average pool -> [group FiLM] -> linear.

    def __init__(self, use_groups=True, n_groups=4, n_portfolios=278):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
                                      nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(32, 1)
        self.film = _film(use_groups, 32, n_groups, n_portfolios)      # created last: same init as without groups

    def forward(self, r, v, g, p=None):
        h = self.features(r).squeeze(-1)
        if self.film is not None:
            h = self.film(h, g, p)
        return self.head(h).squeeze(-1)


def _branch():
    return nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
                         nn.Conv1d(16, 32, 3, padding=1), nn.ReLU())


class ReturnVolumeCNN(nn.Module):
    # Two branches (returns, signed volumes), each pooled to 32 features -> [group FiLM on 64] -> MLP

    def __init__(self, use_groups=True, n_groups=4, n_portfolios=278):
        super().__init__()
        self.return_branch = nn.Sequential(_branch(), nn.AdaptiveAvgPool1d(1))
        self.volume_branch = nn.Sequential(_branch(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.film = _film(use_groups, 64, n_groups, n_portfolios)      # created last: same init as without groups

    def forward(self, r, v, g, p=None):
        h = torch.cat([self.return_branch(r).squeeze(-1), self.volume_branch(v).squeeze(-1)], dim=1)
        if self.film is not None:
            h = self.film(h, g, p)
        return self.head(h).squeeze(-1)


class ReturnCNNNoPool(nn.Module):
    #Return-only CNN without pooling: conv(3) -> conv(3) -> [group FiLM on the channels] -> flatten
    # (32 features x 20 lags) -> MLP. Signed volumes are not used (they led to overfitting)

    def __init__(self, use_groups=True, n_groups=4, n_portfolios=278):
        super().__init__()
        self.return_branch = _branch()
        self.head = nn.Sequential(nn.Linear(32 * len(RET), 32), nn.ReLU(), nn.Linear(32, 1))
        self.film = _film(use_groups, 32, n_groups, n_portfolios)      # created last: same init as without groups

    def forward(self, r, v, g, p=None):
        h = self.return_branch(r)
        if self.film is not None:
            h = self.film(h, g, p)
        return self.head(h.flatten(1)).squeeze(-1)


class MultiScaleReturnCNN(nn.Module):
    # Return-only multi-scale CNN: parallel kernels of size 2, 3 and 5 (10 + 11 + 11 = 32 filters),
    # average pooled -> [group FiLM on 32] -> MLP. Signed volumes are not used (they led to overfitting).

    def __init__(self, use_groups=True, n_groups=4, n_portfolios=278):
        super().__init__()
        self.r2, self.r3, self.r5 = (nn.Conv1d(1, 10, 2, padding=1), nn.Conv1d(1, 11, 3, padding="same"),
                                     nn.Conv1d(1, 11, 5, padding="same"))
        self.relu, self.pool = nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
        self.film = _film(use_groups, 32, n_groups, n_portfolios)      # created last: same init as without groups

    def forward(self, r, v, g, p=None):
        L = r.shape[-1]
        h = torch.cat([self.relu(self.r2(r))[:, :, :L], self.relu(self.r3(r)), self.relu(self.r5(r))], dim=1)
        h = self.pool(h).squeeze(-1)
        if self.film is not None:
            h = self.film(h, g, p)
        return self.head(h).squeeze(-1)


# =============================================================================
# 5. Training and evaluation
# =============================================================================

def _tensors(part):
    return (torch.tensor(part["R"]).unsqueeze(1), torch.tensor(part["V"]).unsqueeze(1),
            torch.tensor(part["g"]), torch.tensor(part["p"]),
            torch.tensor((part["y"] > 0).astype(np.float32)))


def train_one_fold(model_fn, fold, n_epochs=20, batch_size=256, lr=1e-3, weight_decay=0.0,
                   portfolio_weight_decay=1e-3, seed=0):
    # Train a fresh model on one fold; return loss histories, validation predictions and metrics.
    # model_fn: a function with no arguments returning a new model, e.g. lambda: ReturnCNN(use_groups=True).
    # Portfolio embeddings, if present, get their own weight decay (the shrinkage towards the group).
    
    torch.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(*_tensors(fold["train"])), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*_tensors(fold["val"])), batch_size=batch_size, shuffle=False)

    model = model_fn()
    port = [prm for name, prm in model.named_parameters() if "portfolio" in name]
    rest = [prm for name, prm in model.named_parameters() if "portfolio" not in name]
    groups = [{"params": rest, "weight_decay": weight_decay}]
    if port:
        groups.append({"params": port, "weight_decay": portfolio_weight_decay})
    optimizer = torch.optim.AdamW(groups, lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    def run_epoch(loader, train):
        model.train(train)
        total, n, probs = 0.0, 0, []
        with torch.set_grad_enabled(train):
            for r, v, g, p, y in loader:
                logits = model(r, v, g, p)
                loss = criterion(logits, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    probs.append(torch.sigmoid(logits).numpy())
                total += loss.item() * len(y)          # weighted by this batch's own size
                n += len(y)
        return total / n, (np.concatenate(probs) if probs else None)

    train_hist, val_hist = [], []
    for _ in range(n_epochs):
        train_hist.append(run_epoch(train_loader, True)[0])
        val_loss, probabilities = run_epoch(val_loader, False)
        val_hist.append(val_loss)

    y_true = (fold["val"]["y"] > 0).astype(int)
    y_pred = (probabilities >= 0.5).astype(int)
    return {"model": model, "train_loss": train_hist, "val_loss": val_hist,
            "val_idx": fold["val_idx"], "y_true": y_true, "y_pred": y_pred, "probabilities": probabilities,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, probabilities),
            "confusion_matrix": confusion_matrix(y_true, y_pred)}


def run_cv(model_fn, cnn_folds, parallel=True, n_workers=None, verbose=True, **train_kw):
    # Train and evaluate on every fold. With parallel=True the folds run in threads, one CPU thread each.
    n = len(cnn_folds)
    results = [None] * n
    if parallel:
        torch.set_num_threads(1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers or n) as ex:
            futures = {ex.submit(train_one_fold, model_fn, cnn_folds[k], seed=k, **train_kw): k for k in range(n)}
            for fut in concurrent.futures.as_completed(futures):
                k = futures[fut]
                results[k] = fut.result()
                if verbose:
                    print(f"Fold {k}: accuracy = {results[k]['accuracy']:.4f}, "
                          f"balanced = {results[k]['balanced_accuracy']:.4f}, AUC = {results[k]['auc']:.4f}")
    else:
        for k in range(n):
            results[k] = train_one_fold(model_fn, cnn_folds[k], seed=k, **train_kw)
            if verbose:
                print(f"Fold {k}: accuracy = {results[k]['accuracy']:.4f}, "
                      f"balanced = {results[k]['balanced_accuracy']:.4f}, AUC = {results[k]['auc']:.4f}")
    return results


def summarize(results, name=""):
    # Per-fold metrics plus mean and standard deviation over folds
    df = pd.DataFrame({"fold": range(len(results)),
                       "accuracy": [r["accuracy"] for r in results],
                       "balanced_accuracy": [r["balanced_accuracy"] for r in results],
                       "auc": [r["auc"] for r in results]})
    print(f"{name}\n{df.to_string(index=False)}")
    m = df[["accuracy", "balanced_accuracy", "auc"]]
    print(f"mean: {m.mean().round(4).to_dict()}\nstd:  {m.std().round(4).to_dict()}")
    return df


def plot_losses(results, title=""):
    # Training and validation BCE per epoch, one panel per fold
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3), sharey=True)
    for k, (ax, r) in enumerate(zip(np.atleast_1d(axes), results)):
        ep = range(1, len(r["train_loss"]) + 1)
        ax.plot(ep, r["train_loss"], label="training")
        ax.plot(ep, r["val_loss"], label="validation")
        ax.axhline(np.log(2), color="grey", lw=0.8, ls=":")
        ax.set_title(f"{title} fold {k}"); ax.set_xlabel("epoch")
    np.atleast_1d(axes)[0].set_ylabel("BCE loss"); np.atleast_1d(axes)[0].legend(fontsize=8)
    plt.tight_layout(); plt.show()


def oof_probabilities(results, n_rows):
    # Out-of-fold P(up) for every training row, in X_train order
    oof = np.full(n_rows, np.nan)
    for r in results:
        oof[r["val_idx"]] = r["probabilities"]
    return oof


def compare_to_hbm(model_results, reference=HBM_FOLD_ACC):
    # Fold-by-fold paired comparison of each model with the HBM (same folds).
    # model_results: {name: results}. A paired t over 5 folds (4 dof) is a rough guide only
    rows = []
    for name, res in model_results.items():
        acc = np.array([r["accuracy"] for r in res])
        d = (acc - reference) * 100
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        rows.append({"model": name, "CV accuracy": round(acc.mean(), 4), "fold sd": round(acc.std(ddof=1), 4),
                     "vs HBM (pts)": round(d.mean(), 2), "folds better": f"{(d > 0).sum()}/{len(d)}",
                     "paired t": round(t, 1)})
    return pd.DataFrame(rows)
