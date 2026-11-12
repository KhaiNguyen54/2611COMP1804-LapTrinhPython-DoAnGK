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

# Chi dung cac cot co y nghia khi du doan gia; khong dua Address/SellerG vao
# de han che cardinality cao va phu thuoc qua muc vao tung tin dang/nguoi ban.
NUMERIC_CANDIDATES = [
    "Rooms", "Bedroom2", "Bathroom", "Car", "Distance",
    "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude",
    "Propertycount", "Car_was_missing", "BuildingArea_was_missing",
    "YearBuilt_was_missing", "CouncilArea_was_missing", "SaleYear", "SaleMonth",
]
CATEGORICAL_CANDIDATES = [
    "Suburb", "Type", "Method", "Postcode", "CouncilArea", "Regionname",
]


def build_preprocessor(numeric_features, categorical_features):
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


def build_estimator(regressor, numeric_features, categorical_features):
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
        ("model", regressor),
    ])
    # Gia nha lech phai; huan luyen tren log(Price), nhung predict tra ve AUD.
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def evaluate(y_true, y_pred):
    return {
        "MAE_AUD": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_AUD": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay {DATA_PATH}. Hay chay pipeline.py truoc de tao du lieu da lam sach."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise KeyError(f"Dataset khong co cot muc tieu {TARGET!r}.")

    # Loai cac dong khong co gia hop le; khong impute bien muc tieu.
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
    for name, estimator in models.items():
        print(f"Validation {name} ...")
        estimator.fit(X_train, y_train)
        val_prediction = estimator.predict(X_val)
        validation_rows.append({"Model": name, **evaluate(y_val, val_prediction)})

    validation_metrics = pd.DataFrame(validation_rows).sort_values("MAE_AUD").reset_index(drop=True)
    best_name = str(validation_metrics.iloc[0]["Model"])

    # Refitting chi mo hinh da chon tren toan bo tap train.
    # Tap test chi dung de danh gia cuoi cung, khong dung chon mo hinh.
    best_model = models[best_name]
    print(f"Final fit and test evaluation: {best_name} ...")
    best_model.fit(X_train_full, y_train_full)
    best_predictions = best_model.predict(X_test)
    best_metrics = evaluate(y_test, best_predictions)

    baseline = DummyRegressor(strategy="median")
    baseline.fit(X_train_full.iloc[:, [0]], y_train_full)
    baseline_pred = baseline.predict(X_test.iloc[:, [0]])

    metrics = pd.DataFrame([
        {"Model": "Baseline_median", **evaluate(y_test, baseline_pred), "Selected_by_validation": False},
        {"Model": best_name, **best_metrics, "Selected_by_validation": True},
    ]).sort_values("MAE_AUD").reset_index(drop=True)
    validation_metrics = pd.DataFrame(validation_rows).sort_values("MAE_AUD").reset_index(drop=True)
    validation_metrics.to_csv(
        OUTPUT_DIR / "model_validation_metrics.csv", index=False, encoding="utf-8-sig"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "best_pipeline.joblib")
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    test_results = X_test.copy()
    test_results.insert(0, "ActualPrice", y_test.to_numpy())
    test_results["PredictedPrice"] = best_predictions
    test_results["Residual"] = test_results["ActualPrice"] - test_results["PredictedPrice"]
    test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

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

    # Residual distribution.
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
    print(f"\nModel selected by validation MAE: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_pipeline.joblib'}")
    print(f"Saved metadata: {MODEL_DIR / 'model_metadata.json'}")
    print(f"Saved predictions and charts in: {OUTPUT_DIR}")
    print("\nFeatures expected by predict():")
    print(feature_columns)


if __name__ == "__main__":
    main()
