#Unit tests for cnn_utils.py on synthetic data (the challenge data cannot be redistributed).
import numpy as np
import pandas as pd
import pytest
import torch
import cnn_utils as cu

MODELS = [cu.ReturnCNN, cu.ReturnVolumeCNN, cu.ReturnCNNNoPool, cu.MultiScaleReturnCNN]
N_PORTFOLIOS = 40


def synthetic_panel(n_portfolios=N_PORTFOLIOS, n_days=100, seed=0):
    # Overlapping 20-day windows like the challenge data: RET_j on day t is the return of day t-j+1.
    # Dates get shuffled integer labels, so their true order is hidden in TS.
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n_portfolios, n_days + 20)) * 0.01
    labels = rng.permutation(n_days) + 1000
    rows = []
    for a in range(n_portfolios):
        for t in range(n_days):
            row = {"TS": labels[t], "ALLOCATION": f"A{a:03d}", "GROUP": a % 4 + 1, "TRUE_DAY": t,
                   "target": r[a, t + 20] if t + 20 < r.shape[1] else 0.0}
            row.update({f"RET_{j}": r[a, t + 20 - j] for j in range(1, 21)})
            row.update({f"SIGNED_VOLUME_{j}": rng.standard_normal() for j in range(1, 21)})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return cu.reconstruct_dates(synthetic_panel(), verbose=False)


def batch(n=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    r = torch.randn(n, 1, len(cu.RET), generator=gen)
    v = torch.randn(n, 1, len(cu.VOL), generator=gen)
    g = torch.randint(0, 4, (n,), generator=gen)
    p = torch.randint(-1, N_PORTFOLIOS, (n,), generator=gen)      # -1 = portfolio not seen in training
    return r, v, g, p


def test_reconstruct_dates_recovers_true_order(panel):
    assert panel["PIECE"].nunique() == 1
    per_date = panel.groupby("TS")[["POS", "TRUE_DAY"]].first()
    assert (per_date["POS"] == per_date["TRUE_DAY"]).all()          # POS runs forward in time


def test_make_folds_do_not_share_stretches():
    X = synthetic_panel(n_days=30)
    X["PIECE"] = X["TRUE_DAY"] // 5                                  # 6 stretches of 5 days
    folds = cu.make_folds(X, n_splits=3, verbose=False)
    for tr, va in folds:
        assert not set(X["PIECE"].iloc[tr]) & set(X["PIECE"].iloc[va])
    assert sorted(np.concatenate([va for _, va in folds])) == list(range(len(X)))


def test_build_cnn_folds_shapes(panel):
    X = panel.assign(PIECE=panel["POS"] // 10)                         # 10 stretches of 10 days
    folds = cu.make_folds(X, n_splits=2, verbose=False)
    cnn_folds = cu.build_cnn_folds(X, folds, verbose=False)
    for (tr, va), f in zip(folds, cnn_folds):
        assert f["train"]["R"].shape == (len(tr), len(cu.RET))
        assert f["val"]["V"].shape == (len(va), len(cu.VOL))
        assert f["train"]["g"].min() >= 0 and f["train"]["g"].max() < f["n_groups"]
        assert f["n_portfolios"] == N_PORTFOLIOS
        assert np.isfinite(f["train"]["R"]).all() and np.isfinite(f["val"]["V"]).all()


@pytest.mark.parametrize("cls", MODELS)
def test_model_output_shape(cls):
    r, v, g, p = batch()
    model = cls(use_groups=True, n_portfolios=N_PORTFOLIOS)
    assert model(r, v, g, p).shape == (8,)
    assert model(r, v, g).shape == (8,)                              # portfolio index is optional


@pytest.mark.parametrize("cls", MODELS)
def test_untrained_model_equals_model_without_groups(cls):
    # FiLM embeddings start at zero and the layer is created last, so with the same seed
    # the grouped model is exactly the ungrouped one before training.
    r, v, g, p = batch()
    torch.manual_seed(0)
    with_groups = cls(use_groups=True, n_portfolios=N_PORTFOLIOS)
    torch.manual_seed(0)
    without = cls(use_groups=False)
    torch.testing.assert_close(with_groups(r, v, g, p), without(r, v, g, p))


@pytest.mark.parametrize("shape", [(8, 32), (8, 32, 20)])
def test_group_film_shapes(shape):
    film = cu.GroupFiLM(32, n_groups=4, n_portfolios=N_PORTFOLIOS)
    h = torch.randn(*shape)
    _, _, g, p = batch()
    assert film(h, g, p).shape == shape
    torch.testing.assert_close(film(h, g, p), h)                     # identity at initialisation


def synthetic_fold(n_train=2000, n_val=1000, signal=False, seed=0):
    rng = np.random.default_rng(seed)

    def part(n):
        R = rng.standard_normal((n, len(cu.RET))).astype(np.float32)
        y = R[:, 0] if signal else rng.standard_normal(n)            # signal: target = sign of RET_1
        return {"R": R, "V": rng.standard_normal((n, len(cu.VOL))).astype(np.float32),
                "g": rng.integers(0, 4, n), "p": rng.integers(0, N_PORTFOLIOS, n), "y": y}

    return {"train": part(n_train), "val": part(n_val), "val_idx": np.arange(n_val)}


def test_train_one_fold_losses_are_sane():
    # On pure noise the BCE loss must stay near log 2 (guards against mis-weighted batch averages).
    res = cu.train_one_fold(lambda: cu.ReturnCNN(n_portfolios=N_PORTFOLIOS), synthetic_fold(), n_epochs=2)
    assert len(res["train_loss"]) == len(res["val_loss"]) == 2
    assert all(0.6 < l < 0.8 for l in res["train_loss"] + res["val_loss"])
    assert res["probabilities"].shape == (1000,)
    assert 0.0 <= res["accuracy"] <= 1.0


def test_train_one_fold_learns_a_simple_signal():
    res = cu.train_one_fold(lambda: cu.ReturnCNNNoPool(n_portfolios=N_PORTFOLIOS),
                            synthetic_fold(signal=True), n_epochs=5)
    assert res["accuracy"] > 0.8


def test_run_cv_parallel_matches_folds():
    folds = [synthetic_fold(n_train=500, n_val=200, seed=k) for k in range(2)]
    results = cu.run_cv(lambda: cu.ReturnCNN(n_portfolios=N_PORTFOLIOS), folds, verbose=False, n_epochs=1)
    assert len(results) == 2 and all(r["probabilities"].shape == (200,) for r in results)
