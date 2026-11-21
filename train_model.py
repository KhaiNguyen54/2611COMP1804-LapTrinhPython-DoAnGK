from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "outputs" / "melb_data_clean.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET = "Price"
RANDOM_STATE = 42

# Biến số 
NUMERIC_CANDIDATES = [
    "Rooms", "Bedroom2", "Bathroom", "Car", "Distance",
    "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude",
    "Propertycount", "Car_was_missing", "BuildingArea_was_missing",
    "YearBuilt_was_missing", "CouncilArea_was_missing", "SaleYear", "SaleMonth",
]
# Biến phân loại
CATEGORICAL_CANDIDATES = [
    "Suburb", "Type", "Method", "Postcode", "CouncilArea", "Regionname",
]

# Hàm tiền xử lý dữ liệu
def build_preprocessor(numeric_features, categorical_features): # 2 tham số truyền 
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

# Hàm xây dựng mô hình ước lượng với tiền xử lý và biến mục tiêu được biến đổi
def build_estimator(regressor, numeric_features, categorical_features):
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("model", regressor),
    ])

    # Giá nhà lệch phải, train theo log1p(Price) và dự đoán ngược lại bằng expm1.   
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

# Hàm đánh giá mô hình qua các chỉ số (MAE, RMSE, R2)
def evaluate(y_true, y_pred):
    return {
        "MAE_AUD": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_AUD": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# Một dòng của bảng kiểm tra quá khớp / dưới khớp: sai số trên tập train và trên tập chưa dùng để học.
def build_fit_check_row(model_name, eval_set, train_metrics, eval_metrics):
    return {
        "Model": model_name,
        "Eval_set": eval_set,
        "MAE_train": train_metrics["MAE_AUD"],
        "MAE_eval": eval_metrics["MAE_AUD"],
        "RMSE_train": train_metrics["RMSE_AUD"],
        "RMSE_eval": eval_metrics["RMSE_AUD"],
        "R2_train": train_metrics["R2"],
        "R2_eval": eval_metrics["R2"],
    }


# Gộp các dòng thành bảng và thêm tỷ lệ MAE_eval / MAE_train (gần 1: hai tập tương đương; lớn: có khoảng cách).
def build_fit_check(rows):
    fit_check = pd.DataFrame(rows)
    fit_check["MAE_eval_chia_MAE_train"] = fit_check["MAE_eval"] / fit_check["MAE_train"]
    return fit_check

# Hàm in kết quả fit check
def print_fit_check(fit_check):
    print("\n=== Kiem tra Overfitting / Underfitting (sai so train so voi tap chua dung de hoc) ===")
    print(fit_check.to_string(index=False, formatters={
        "MAE_train": "{:,.0f}".format,
        "MAE_eval": "{:,.0f}".format,
        "RMSE_train": "{:,.0f}".format,
        "RMSE_eval": "{:,.0f}".format,
        "R2_train": "{:.4f}".format,
        "R2_eval": "{:.4f}".format,
        "MAE_eval_chia_MAE_train": "{:.2f}".format,
    }))



# Hàm tổng hợp số đo sai số mà báo cáo trích (nhóm M); chỉ đọc kết quả đã có, không huấn luyện thêm
def build_model_error_metrics(validation_metrics, metrics, test_results, y_train, y_val, y_test, y_train_full):
    rows = []

    def r(ma, noi_dung, gia_tri, cach_tinh):
        if isinstance(gia_tri, (float, np.floating)):
            gia_tri = round(float(gia_tri), 4)
        rows.append({"ma": ma, "noi_dung": noi_dung, "gia_tri": gia_tri, "cach_tinh": cach_tinh})

    n_total = len(y_train_full) + len(y_test)
    for ten, s in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        r("M06", f"Tập {ten}: số dòng / tỷ lệ / giá trung vị", f"{len(s)} / {len(s) / n_total * 100:.0f}% / {s.median():.0f}", "train_test_split(test_size=0.2, random_state=42) hai lần")
    r("M06", "Train gộp validation (dùng huấn luyện lại)", len(y_train_full), "train_test_split lần 1")

    vm = validation_metrics.set_index("Model")
    tm = metrics.set_index("Model")
    r("M01", "MAE validation của Random Forest thấp hơn Ridge (%)", (1 - vm.loc["RandomForest_log_target", "MAE_AUD"] / vm.loc["Ridge", "MAE_AUD"]) * 100, "outputs/model_validation_metrics.csv")
    r("M01", "MAE test của Random Forest thấp hơn baseline (%)", (1 - tm.loc["RandomForest_log_target", "MAE_AUD"] / tm.loc["Baseline_median", "MAE_AUD"]) * 100, "outputs/model_metrics.csv")
    t = test_results
    a, p = t.ActualPrice, t.PredictedPrice
    e = a - p
    ape = e.abs() / a
    r("M02", "Số dòng test; hệ số tương quan giữa giá thực và giá dự đoán", f"{len(t)}; {np.corrcoef(a, p)[0, 1]:.3f}", "outputs/test_predictions.csv")
    r("M02", "MAPE; sai số tương đối trung vị (%)", f"{ape.mean() * 100:.1f}; {ape.median() * 100:.1f}", "|thực - dự đoán| / thực")
    r("M02", "Tỷ lệ lệch không quá 10%; 20% (%)", f"{(ape <= .1).mean() * 100:.1f}; {(ape <= .2).mean() * 100:.1f}", "|thực - dự đoán| / thực")
    r("M02", "Sai số (thực - dự đoán): trung vị; trung bình", f"{e.median():.0f}; {e.mean():.0f}", "test_results['Residual']")
    hi = a > 1.5e6
    r("M02", "Giá thực trên 1.5 triệu: số dòng; % bị dự đoán thấp", f"{int(hi.sum())}; {(p[hi] < a[hi]).mean() * 100:.0f}", "outputs/test_predictions.csv")
    r("M02", "Giá thực lớn nhất; giá dự đoán lớn nhất", f"{a.max():.0f}; {p.max():.0f}", "outputs/test_predictions.csv")
    seg = pd.cut(a, [0, 6e5, 1e6, 1.5e6, 1e12], right=False, labels=["<600k", "600k-1M", "1M-1.5M", ">=1.5M"])
    for k, gdf in t.assign(seg=seg, ae=e.abs(), ape=ape * 100).groupby("seg", observed=True):
        r("M03", f"Khoảng giá {k}: số dòng; MAE; MAPE (%); lệch trung bình", f"{len(gdf)}; {gdf.ae.mean():.0f}; {gdf.ape.mean():.1f}; {(gdf.ActualPrice - gdf.PredictedPrice).mean():+.0f}", "outputs/test_predictions.csv")
    for k, gdf in t.assign(ae=e.abs(), ape=ape * 100).groupby("Type"):
        r("M03", f"Loại nhà {k}: số dòng; MAE; MAPE (%); lệch trung bình", f"{len(gdf)}; {gdf.ae.mean():.0f}; {gdf.ape.mean():.1f}; {(gdf.ActualPrice - gdf.PredictedPrice).mean():+.0f}", "outputs/test_predictions.csv")
    for k, gdf in t.assign(ae=e.abs()).groupby("Regionname"):
        r("M04", f"Vùng {k}: số dòng test; MAE", f"{len(gdf)}; {gdf.ae.mean():.0f}", "outputs/test_predictions.csv")
    top = t.assign(ae=e.abs(), e=e).sort_values("ae", ascending=False)
    for _, x in top.head(3).iterrows():
        r("M05", "Ba sai số lớn nhất (dự đoán thấp hơn thực tế): suburb, loại, giá thực, giá dự đoán", f"{x.Suburb}, {x.Type}, {x.ActualPrice:.0f}, {x.PredictedPrice:.0f}", "outputs/test_predictions.csv")
    x = top[top.e < 0].iloc[0]
    r("M05", "Sai số lớn nhất theo chiều ngược lại: suburb, loại, giá thực, giá dự đoán", f"{x.Suburb}, {x.Type}, {x.ActualPrice:.0f}, {x.PredictedPrice:.0f}", "outputs/test_predictions.csv")

    return pd.DataFrame(rows)



def save_charts(y_test, best_predictions, best_name):
    # Actual vs predicted chart.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, best_predictions, alpha=0.35, s=18)
    lower = min(float(y_test.min()), float(np.min(best_predictions)))
    upper = max(float(y_test.max()), float(np.max(best_predictions)))
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="red")
    ax.set_title(f"Actual vs Predicted — {best_name}")
    ax.set_xlabel("Actual Price (AUD)")
    ax.set_ylabel("Predicted Price (AUD)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Residual distribution chart 
    residuals = y_test.to_numpy() - best_predictions
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=40, edgecolor="white")
    ax.axvline(0, linestyle="--", color="red")
    ax.set_title(f"Residual Distribution — {best_name}")
    ax.set_xlabel("Residual (Actual - Predicted), AUD")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_residual_distribution.png", dpi=160)
    plt.close(fig)

# Hàm main huấn luyện model
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay {DATA_PATH}. Hay chay pipeline.py truoc de tao du lieu da lam sach."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise KeyError(f"Dataset khong co cot muc tieu {TARGET!r}.")

    # Loại bỏ các giá trị không hợp lệ, không impute biến mục tiêu, giữ Price > 0.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.loc[df[TARGET].notna() & (df[TARGET] > 0)].copy()

    numeric_features = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    feature_columns = numeric_features + categorical_features

    if not feature_columns:
        raise ValueError("Khong tim thay cot dau vao phu hop de huan luyen.")

    X = df[feature_columns].copy()
    y = df[TARGET].astype(float).copy()

    # Chia test rieng ngay tu dau; validation chi dung de chon mo hinh.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Ridge": Pipeline([
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", Ridge(alpha=10.0, solver="lsqr")),
        ]),
        "RandomForest_log_target": build_estimator(
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=2,
                max_features=0.9,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            numeric_features,
            categorical_features,
        ),
    }

    # Chon mo hinh bang validation MAE, khong chon theo tap test.
    validation_rows = []
    fit_check_rows = []
    for name, estimator in models.items():
        print(f"Validation {name} ...")
        estimator.fit(X_train, y_train)
        val_prediction = estimator.predict(X_val)
        val_metrics = evaluate(y_val, val_prediction)
        validation_rows.append({"Model": name, **val_metrics})
        train_metrics = evaluate(y_train, estimator.predict(X_train))
        fit_check_rows.append(build_fit_check_row(name, "validation", train_metrics, val_metrics))

    validation_metrics = pd.DataFrame(validation_rows).sort_values("MAE_AUD").reset_index(drop=True)
    best_name = str(validation_metrics.iloc[0]["Model"])

    # Refitting chi mo hinh da chon tren toan bo tap train.
    # Tap test chi dung de danh gia cuoi cung, khong dung chon mo hinh.
    best_model = models[best_name]
    print(f"Final fit and test evaluation: {best_name} ...")
    best_model.fit(X_train_full, y_train_full)
    best_predictions = best_model.predict(X_test)
    best_metrics = evaluate(y_test, best_predictions)
    final_train_metrics = evaluate(y_train_full, best_model.predict(X_train_full))
    fit_check_rows.append(build_fit_check_row(
        f"{best_name} (huấn luyện lại trên train gộp validation)", "test", final_train_metrics, best_metrics
    ))

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train_full.iloc[:, [0]], y_train_full)
    baseline_pred = baseline.predict(X_test.iloc[:, [0]])

    metrics = pd.DataFrame([
        {"Model": "Baseline_median", **evaluate(y_test, baseline_pred), "Selected_by_validation": False},
        {"Model": best_name, **best_metrics, "Selected_by_validation": True},
    ]).sort_values("MAE_AUD").reset_index(drop=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_metrics.to_csv(
        OUTPUT_DIR / "model_validation_metrics.csv", index=False, encoding="utf-8-sig"
    )

    fit_check = build_fit_check(fit_check_rows)
    fit_check.to_csv(OUTPUT_DIR / "model_fit_check.csv", index=False, encoding="utf-8-sig")

    joblib.dump(best_model, MODEL_DIR / "best_pipeline.joblib")
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    test_results = X_test.copy()
    test_results.insert(0, "ActualPrice", y_test.to_numpy())
    test_results["PredictedPrice"] = best_predictions
    test_results["Residual"] = test_results["ActualPrice"] - test_results["PredictedPrice"]
    test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    build_model_error_metrics(
        validation_metrics, metrics, test_results, y_train, y_val, y_test, y_train_full
    ).to_csv(OUTPUT_DIR / "model_error_metrics.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "target": TARGET,
        "best_model": best_name,
        "best_metrics": best_metrics,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": feature_columns,
        "train_rows": int(len(X_train_full)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "target_transform": "Random Forest trains on log1p(Price) and inverse-transforms predictions; Ridge predicts raw Price; all metrics are in AUD",
        "selection_method": "Lowest validation MAE; final evaluation uses a separate 20% test set",
        "dataset": str(DATA_PATH.relative_to(PROJECT_DIR)),
        "note": "Models were selected on a validation split; a separate test split is used for final evaluation. Use a time-based split for strict future-period forecasting.",
    }
    with (MODEL_DIR / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    save_charts(y_test, best_predictions, best_name)

    print("\n=== Model selection (validation set) ===")
    print(validation_metrics.to_string(index=False, formatters={
        "MAE_AUD": "{:,.0f}".format,
        "RMSE_AUD": "{:,.0f}".format,
        "R2": "{:.4f}".format,
    }))
    print("\n=== Final model comparison (test set) ===")
    print(metrics.to_string(index=False, formatters={
        "MAE_AUD": "{:,.0f}".format,
        "RMSE_AUD": "{:,.0f}".format,
        "R2": "{:.4f}".format,
    }))
    print_fit_check(fit_check)
    print(f"\nModel selected by validation MAE: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_pipeline.joblib'}")
    print(f"Saved metadata: {MODEL_DIR / 'model_metadata.json'}")
    print(f"Saved predictions and charts in: {OUTPUT_DIR}")
    print("\nFeatures expected by predict():")
    print(feature_columns)


if __name__ == "__main__":
    main()
  