"""Sinh các hình dùng trong báo cáo từ dữ liệu thật, đưa vào report/figures/.

Chạy từ thư mục report/:  python make_figures.py

- Năm biểu đồ phân tích (01-05) được vẽ bằng đúng hàm create_charts() của pipeline.py.
- Hai biểu đồ chẩn đoán mô hình (06-07) được vẽ từ outputs/test_predictions.csv
  (kết quả của train_model.py), cùng kiểu với biểu đồ train_model.py tạo ra.
- Hai biểu đồ chất lượng dữ liệu (f_missing, f_cleaning) được vẽ từ tệp gốc data/melb_data.csv.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import pipeline as P

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
SIZE = (6.4, 3.9)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("đã tạo", name)


def f_missing(raw):
    sns.set_theme(style="whitegrid")
    m = raw.isna().mean() * 100
    m = m[m > 0].sort_values()
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.barh(m.index, m.values, color="#4f81bd")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlim(0, 58)
    ax.set_xlabel(f"Tỷ lệ dữ liệu thiếu (% trên {len(raw):,} bản ghi)")
    ax.set_ylabel("Cột")
    ax.set_title("Dữ liệu thiếu trong tệp gốc")
    save(fig, "f_missing.png")


def f_cleaning(raw):
    """Số dòng bị ảnh hưởng bởi từng quy tắc làm sạch của pipeline.clean_data()."""
    sns.set_theme(style="whitegrid")
    rules = {
        "CouncilArea thiếu → Unknown": int(raw["CouncilArea"].isna().sum()),
        "BuildingArea ≤ 0 → thiếu": int((raw["BuildingArea"] <= 0).sum()),
        "YearBuilt < 1700 → thiếu": int((raw["YearBuilt"] < 1700).sum()),
        "Bathroom = 0 → thiếu": int((raw["Bathroom"] == 0).sum()),
        "Landsize > 10,000 → thiếu": int((raw["Landsize"] > P.MAX_LANDSIZE).sum()),
        "BuildingArea > 1,000 → thiếu": int((raw["BuildingArea"] > P.MAX_BUILDING_AREA).sum()),
        "Price ≤ 0 (bỏ dòng)": int((raw["Price"] <= 0).sum()),
        "Rooms ≤ 0 → thiếu": int((raw["Rooms"] <= 0).sum()),
        "Distance < 0 → thiếu": int((raw["Distance"] < 0).sum()),
        "Landsize < 0 → thiếu": int((raw["Landsize"] < 0).sum()),
        "Tọa độ ngoài vùng → thiếu": int(
            (~raw["Lattitude"].between(-39.5, -36.0)).sum() + (~raw["Longtitude"].between(142.0, 147.0)).sum()
        ),
        "Dòng trùng hoàn toàn (bỏ dòng)": int(raw.duplicated().sum()),
    }
    s = pd.Series(rules).sort_values()
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.barh(s.index, s.values, color="#c0504d")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlim(0, 1600)
    ax.set_xlabel("Số dòng bị ảnh hưởng")
    ax.set_ylabel("Quy tắc làm sạch")
    ax.set_title("Số dòng bị ảnh hưởng theo từng quy tắc")
    save(fig, "f_cleaning.png")


def model_charts():
    """Vẽ lại hai biểu đồ chẩn đoán của train_model.py từ outputs/test_predictions.csv."""
    path = ROOT / "outputs" / "test_predictions.csv"
    if not path.exists():
        print("bỏ qua biểu đồ mô hình: chưa có", path, "(chạy train_model.py trước)")
        return
    t = pd.read_csv(path)
    y, pred = t["ActualPrice"].to_numpy(), t["PredictedPrice"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y, pred, alpha=0.35, s=18)
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="red")
    ax.set_title("Giá thực tế và giá dự đoán trên tập test")
    ax.set_xlabel("Giá thực tế (AUD)")
    ax.set_ylabel("Giá dự đoán (AUD)")
    save(fig, "06_actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y - pred, bins=40, edgecolor="white")
    ax.axvline(0, linestyle="--", color="red")
    ax.set_title("Phân phối sai số (thực tế − dự đoán) trên tập test")
    ax.set_xlabel("Sai số (AUD)")
    ax.set_ylabel("Số bản ghi")
    save(fig, "07_residual_distribution.png")


def main():
    raw = P.load_data()
    clean = P.clean_data(raw)
    for p in P.create_charts(clean, OUT):
        print("đã tạo", p.name)
    f_missing(raw)
    f_cleaning(raw)
    model_charts()


if __name__ == "__main__":
    main()
