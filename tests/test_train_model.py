"""Kiểm thử các hàm của train_model.py (không huấn luyện mô hình để chạy nhanh)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import train_model as tm

OUT = tm.OUTPUT_DIR


def test_split_sizes_are_64_16_20():
    idx = np.arange(13_580)
    full, test = train_test_split(idx, test_size=0.2, random_state=tm.RANDOM_STATE)
    train, val = train_test_split(full, test_size=0.2, random_state=tm.RANDOM_STATE)
    assert (len(train), len(val), len(test)) == (8_691, 2_173, 2_716)
    assert len(set(train) | set(val) | set(test)) == 13_580  # ba tập không giao nhau


def test_evaluate_perfect_and_known_values():
    y = np.array([100.0, 200.0, 300.0])
    perfect = tm.evaluate(y, y)
    assert perfect == {"MAE_AUD": 0.0, "RMSE_AUD": 0.0, "R2": 1.0}
    m = tm.evaluate(y, y + 10)
    assert m["MAE_AUD"] == pytest.approx(10.0)
    assert m["RMSE_AUD"] == pytest.approx(10.0)


def test_evaluate_median_baseline_has_non_positive_r2():
    y = np.array([1.0, 2.0, 3.0, 100.0])
    pred = np.full_like(y, np.median(y))
    assert tm.evaluate(y, pred)["R2"] <= 0


def test_build_fit_check_ratio_and_columns():
    train = {"MAE_AUD": 100.0, "RMSE_AUD": 150.0, "R2": 0.9}
    ev = {"MAE_AUD": 250.0, "RMSE_AUD": 400.0, "R2": 0.7}
    rows = [tm.build_fit_check_row("M", "validation", train, ev)]
    table = tm.build_fit_check(rows)
    assert table.loc[0, "MAE_eval_chia_MAE_train"] == pytest.approx(2.5)
    assert list(table.columns) == [
        "Model", "Eval_set", "MAE_train", "MAE_eval", "RMSE_train", "RMSE_eval",
        "R2_train", "R2_eval", "MAE_eval_chia_MAE_train",
    ]


def test_print_fit_check_runs(capsys):
    row = tm.build_fit_check_row(
        "M", "test",
        {"MAE_AUD": 1.0, "RMSE_AUD": 2.0, "R2": 0.9},
        {"MAE_AUD": 2.0, "RMSE_AUD": 3.0, "R2": 0.8},
    )
    tm.print_fit_check(tm.build_fit_check([row]))
    assert "Overfitting / Underfitting" in capsys.readouterr().out


def test_preprocessor_has_expected_branches():
    pre = tm.build_preprocessor(["Rooms"], ["Type"])
    assert [name for name, _, _ in pre.transformers] == ["numeric", "categorical"]
    assert pre.remainder == "drop"


def test_feature_lists_have_no_target_or_overlap():
    assert tm.TARGET not in tm.NUMERIC_CANDIDATES + tm.CATEGORICAL_CANDIDATES
    assert not set(tm.NUMERIC_CANDIDATES) & set(tm.CATEGORICAL_CANDIDATES)
    assert len(tm.NUMERIC_CANDIDATES) + len(tm.CATEGORICAL_CANDIDATES) == 23


def test_committed_fit_check_is_consistent():
    table = pd.read_csv(OUT / "model_fit_check.csv", encoding="utf-8-sig")
    assert len(table) == 3
    ratio = table["MAE_eval"] / table["MAE_train"]
    np.testing.assert_allclose(ratio, table["MAE_eval_chia_MAE_train"])
    ridge = table[table["Model"] == "Ridge"].iloc[0]
    assert ridge["MAE_eval_chia_MAE_train"] < 1.1  # dưới khớp: hai sai số gần bằng nhau


def test_committed_validation_matches_fit_check():
    val = pd.read_csv(OUT / "model_validation_metrics.csv", encoding="utf-8-sig").set_index("Model")
    fit = pd.read_csv(OUT / "model_fit_check.csv", encoding="utf-8-sig")
    fit = fit[fit["Eval_set"] == "validation"].set_index("Model")
    for name in fit.index:
        # Ridge lệch khoảng 0.01% giữa các phiên bản thư viện (231,160 so với 231,137) nên cho phép sai số tương đối 0.1%.
        assert val.loc[name, "MAE_AUD"] == pytest.approx(fit.loc[name, "MAE_eval"], rel=1e-3)
