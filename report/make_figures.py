"""Sinh các hình dùng trong báo cáo từ dữ liệu thật (data/melb_data.csv), đưa vào report/figures/.
Chạy từ thư mục report/:  python make_figures.py
Các biểu đồ phân tích dùng lại đúng hàm trong ml_pipeline/analysis.py, nên giống hệt biểu đồ trong ứng dụng Streamlit."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml_pipeline"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import analysis as A
import features as F

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
    m = (raw.isna().mean() * 100)
    m = m[m > 0].sort_values()
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.barh(m.index, m.values, color="#4f81bd")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlim(0, 58)
    ax.set_xlabel("Tỷ lệ dữ liệu thiếu (% trên 13,580 bản ghi)")
    ax.set_ylabel("Cột")
    ax.set_title("Dữ liệu thiếu trong tệp gốc")
    save(fig, "f_missing.png")


def f_cleaning(raw):
    sns.set_theme(style="whitegrid")
    rules = {
        "Landsize = 0": int((raw.Landsize == 0).sum()),
        "CouncilArea thiếu": int(raw.CouncilArea.isna().sum()),
        "Bathroom = 0": int((raw.Bathroom == 0).sum()),
        "BuildingArea = 0": int((raw.BuildingArea == 0).sum()),
        "YearBuilt < 1800": int((raw.YearBuilt < 1800).sum()),
        "Landsize > 100,000": int((raw.Landsize > 100000).sum()),
        "BuildingArea > 10,000": int((raw.BuildingArea > 10000).sum()),
    }
    s = pd.Series(rules).sort_values()
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.barh(s.index, s.values, color="#c0504d")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlim(0, 2300)
    ax.set_xlabel("Số dòng bị ảnh hưởng")
    ax.set_ylabel("Quy tắc làm sạch")
    ax.set_title("Số dòng bị ảnh hưởng theo từng quy tắc")
    save(fig, "f_cleaning.png")


def f_split(d):
    sns.set_theme(style="whitegrid")
    d = d.sort_values("Date")
    d_tr, d_va = d["Date"].quantile(0.70), d["Date"].quantile(0.85)
    parts = [("Train", d[d.Date <= d_tr], "#4f81bd"), ("Validation", d[(d.Date > d_tr) & (d.Date <= d_va)], "#f79646"),
             ("Test", d[d.Date > d_va], "#c0504d")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, p, c in parts:
        m = p.groupby(p.Date.dt.to_period("M")).size()
        ax.bar(m.index.to_timestamp(), m.values, width=20, color=c, label=f"{name} ({len(p):,} dòng)")
    ax.set_xlabel("Tháng bán")
    ax.set_ylabel("Số giao dịch")
    ax.set_title("Chia dữ liệu theo thời gian: train < validation < test")
    ax.legend()
    save(fig, "f_split.png")


def f_price_hist(d):
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.6))
    sns.histplot(d["Price"] / 1e6, bins=60, ax=axs[0], color="teal")
    axs[0].set_xlabel("Giá (triệu AUD)"); axs[0].set_ylabel("Số giao dịch")
    axs[0].set_title(f"Giá gốc (độ lệch {d.Price.skew():.2f})", fontsize=10)
    sns.histplot(np.log(d["Price"]), bins=60, ax=axs[1], color="#7f6084")
    axs[1].set_xlabel("log(Giá), Giá tính bằng AUD"); axs[1].set_ylabel("Số giao dịch")
    axs[1].set_title(f"Sau khi lấy log (độ lệch {np.log(d.Price).skew():.2f})", fontsize=10)
    save(fig, "f_price_hist.png")


def f_rooms(d):
    sns.set_theme(style="whitegrid")
    t = A.stats_by_rooms(d)
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.bar(t.index.astype(str), t["gia_trung_vi"] / 1e6, color="#4f81bd")
    ax.bar_label(bars, labels=[f"n={n:,}" for n in t["so_giao_dich"]], fontsize=8, padding=2)
    ax.set_xlabel("Số phòng"); ax.set_ylabel("Giá trung vị (triệu AUD)")
    ax.set_title("Giá trung vị theo số phòng (n = số giao dịch)")
    ax.set_ylim(0, 2.1)
    save(fig, "f_rooms.png")


def main():
    raw = pd.read_csv(A.DATA)
    d = A.load_clean()
    f_missing(raw)
    f_cleaning(raw)
    f_split(d)
    f_price_hist(d)
    f_rooms(d)
    save(A.fig_type_box(d), "f_type_box.png")
    save(A.fig_trend(d), "f_trend.png")
    save(A.fig_corr(d), "f_corr.png")
    save(A.fig_distance_scatter(d), "f_distance.png")
    save(A.fig_region_bar(d), "f_region.png")


if __name__ == "__main__":
    main()
