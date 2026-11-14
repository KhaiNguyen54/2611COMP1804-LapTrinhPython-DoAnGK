from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "melb_data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Constants 
MAX_LANDSIZE = 10_000       # m2 
MAX_BUILDING_AREA = 1_000   # m2: 
sns.set_theme(style="whitegrid")

# Hàm kiểm tra đường dẫn data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. A fresh DataFrame is returned for each run."""
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset tại: {path}\n"
            "Hãy đặt melb_data.csv trong thư mục data/ của dự án."
        )
    return pd.read_csv(path)

# Hàm kiểm tra data
def audit_data(df: pd.DataFrame) -> dict:
    """Create structured quality reports without modifying the input data."""
    profile = pd.DataFrame({
        "Kiểu data": df.dtypes.astype(str),
        "số lượng missing": df.isna().sum(),
        "% missing": (df.isna().mean() * 100).round(2),
        "giá trị unique": df.nunique(dropna=True),
    }).sort_values("% missing", ascending=False)

# Đảm bảo các cột số liệu được kiểm tra 
    numeric_cols = df.select_dtypes(include=np.number).columns
    invalid = {}
    if "Price" in df.columns:
        invalid["Price ≤ 0"] = int((df["Price"] <= 0).sum())
    if "Rooms" in df.columns:
        invalid["Rooms ≤ 0"] = int((df["Rooms"] <= 0).sum())
    if "Distance" in df.columns:
        invalid["Distance < 0"] = int((df["Distance"] < 0).sum())
    if "BuildingArea" in df.columns:
        invalid["BuildingArea ≤ 0"] = int((df["BuildingArea"] <= 0).sum())
    if "YearBuilt" in df.columns:
        invalid["YearBuilt < 1700"] = int((df["YearBuilt"] < 1700).sum())
    if "Landsize" in df.columns:
        invalid[f"Landsize > {MAX_LANDSIZE}"] = int((df["Landsize"] > MAX_LANDSIZE).sum())
    if "BuildingArea" in df.columns:
        invalid[f"BuildingArea > {MAX_BUILDING_AREA}"] = int((df["BuildingArea"] > MAX_BUILDING_AREA).sum())
    if "Bathroom" in df.columns:
        invalid["Bathroom == 0"] = int((df["Bathroom"] == 0).sum())

    return {
        "shape": df.shape,
        "profile": profile,
        "duplicates": int(df.duplicated().sum()),
        "invalid_counts": pd.Series(invalid, name="count"),
        "numeric_summary": df.select_dtypes(include=np.number).describe().T,
    }

# Hàm làm sạch data 
def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = df.drop_duplicates().copy()

# Xử lý cột ngày tháng
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# Đảm bảo các cột số là kiểu số, chuyển đổi các giá trị không hợp lệ thành NaN
    numeric_cols = [
        "Rooms", "Price", "Distance", "Postcode", "Bedroom2", "Bathroom",
        "Car", "Landsize", "BuildingArea", "YearBuilt", "Lattitude",
        "Longtitude", "Propertycount"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Giữ nguyên thông tin về các giá trị bị thiếu trước khi loại bỏ các giá trị không hợp lý
    for col in ["Car", "BuildingArea", "YearBuilt", "CouncilArea"]:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype("int8")

    # Loại bỏ các giá trị không hợp lý theo quy tắc đã xác định
    if "Price" in df.columns:
        df.loc[df["Price"] <= 0, "Price"] = np.nan
    if "Rooms" in df.columns:
        df.loc[df["Rooms"] <= 0, "Rooms"] = np.nan
    if "Distance" in df.columns:
        df.loc[df["Distance"] < 0, "Distance"] = np.nan
    if "Landsize" in df.columns:
        df.loc[df["Landsize"] < 0, "Landsize"] = np.nan
    if "BuildingArea" in df.columns:
        df.loc[df["BuildingArea"] <= 0, "BuildingArea"] = np.nan
    if "YearBuilt" in df.columns:
        df.loc[df["YearBuilt"] < 1700, "YearBuilt"] = np.nan
    if "Landsize" in df.columns:
        df.loc[df["Landsize"] > MAX_LANDSIZE, "Landsize"] = np.nan
    if "BuildingArea" in df.columns:
        df.loc[df["BuildingArea"] > MAX_BUILDING_AREA, "BuildingArea"] = np.nan
    if "Bathroom" in df.columns:
        df.loc[df["Bathroom"] == 0, "Bathroom"] = np.nan
    if "Lattitude" in df.columns:
        df.loc[~df["Lattitude"].between(-39.5, -36.0), "Lattitude"] = np.nan
    if "Longtitude" in df.columns:
        df.loc[~df["Longtitude"].between(142.0, 147.0), "Longtitude"] = np.nan

    # Giữ Price vì đây là biến mục tiêu
    if "Price" in df.columns:
        df = df.dropna(subset=["Price"])

    # Điền giá trị Unknown cho cột CouncilArea nếu bị thiếu
    if "CouncilArea" in df.columns:
        df["CouncilArea"] = df["CouncilArea"].fillna("Unknown")

    # Tách cột ngày tháng thành các cột riêng biệt cho năm và tháng
    # Lí do tách: để phân tích xu hướng giá theo thời gian xác định
    if "Date" in df.columns:
        df["SaleYear"] = df["Date"].dt.year
        df["SaleMonth"] = df["Date"].dt.month

    return df.reset_index(drop=True)

# Hàm phân tích dữ liệu (Thống kê mô tả, groupby nhóm và tương quan)
def analyze_data(df: pd.DataFrame) -> dict:
    analyses = {
        "price_summary": df["Price"].describe().to_frame("Price"),
        "price_by_type": df.groupby("Type", dropna=False)["Price"].agg(
            count="count", mean="mean", median="median", std="std"
        ).sort_values("median", ascending=False),
        "price_by_region": df.groupby("Regionname", dropna=False)["Price"].agg(
            count="count", mean="mean", median="median"
        ).sort_values("median", ascending=False),
        "price_by_rooms": df.groupby("Rooms", dropna=False)["Price"].agg(
            count="count", mean="mean", median="median"
        ).sort_index(),
        "top_suburbs": df.groupby("Suburb")["Price"].agg(
            count="count", median="median", mean="mean"
        ).query("count >= 30").sort_values("median", ascending=False).head(10),
        "price_distance_corr": df[["Price", "Distance", "Rooms", "BuildingArea", "Landsize"]].corr(numeric_only=True),
        "price_by_sale_month": df.groupby("SaleMonth")["Price"].agg(
            count="count", median="median", mean="mean"
        ).sort_index() if "SaleMonth" in df.columns else pd.DataFrame(),
    }
    return analyses

# Hàm tạo biểu đồ trực quan hóa dữ liệu
def create_charts(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Histogram: phân phối giá bán nhà.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Price"].dropna(), bins=45, kde=True, color=sns.color_palette()[0], ax=ax)
    ax.set(title="Phân phối giá bán nhà tại Melbourne", xlabel="Giá bán (AUD)", ylabel="Số bất động sản")
    fig.tight_layout()
    path = output_dir / "01_price_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 2. Boxplot: giá bán theo loại bất động sản (Type).
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df.groupby("Type")["Price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Type", y="Price", order=order, showfliers=False, color=sns.color_palette()[0], ax=ax)
    ax.set(title="So sánh giá bán theo loại bất động sản", xlabel="Loại (h: house, u: unit, t: townhouse)", ylabel="Giá bán (AUD)")
    fig.tight_layout()
    path = output_dir / "02_price_by_type_boxplot.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 3. Bar chart: trung vị giá bán theo khu vực (Regionname).
    region_median = df.groupby("Regionname")["Price"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=region_median.values, y=region_median.index, color=sns.color_palette()[0], ax=ax)
    ax.set(title="Giá bán trung vị theo khu vực", xlabel="Giá bán trung vị (AUD)", ylabel="Khu vực")
    fig.tight_layout()
    path = output_dir / "03_median_price_by_region.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 4. Scatter plot: khoảng cách đến CBD và giá bán. 
    scatter_df = df[["Distance", "Price"]].dropna()
    if len(scatter_df) > 4000:
        scatter_df = scatter_df.sample(4000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=scatter_df, x="Distance", y="Price", alpha=0.35, s=25, color=sns.color_palette()[0], ax=ax)
    ax.set(title="Khoảng cách đến CBD và giá bán", xlabel="Khoảng cách đến CBD (km)", ylabel="Giá bán (AUD)")
    fig.tight_layout()
    path = output_dir / "04_price_vs_distance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    # 5. Correlation heatmap: Tương quan giữa các biến số
    corr_cols = [c for c in ["Price", "Rooms", "Bedroom2", "Bathroom", "Car", "Distance", "Landsize", "BuildingArea", "YearBuilt", "Propertycount"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Ma trận tương quan giữa các thuộc tính số")
    fig.tight_layout()
    path = output_dir / "05_correlation_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    return saved

# Hàm xây dựng các phát hiện từ dữ liệu
def build_findings(df: pd.DataFrame) -> list[str]:
    findings = []
    median_by_type = df.groupby("Type")["Price"].median().dropna().sort_values(ascending=False)
    if len(median_by_type) >= 2:
        high_type, low_type = median_by_type.index[0], median_by_type.index[-1]
        ratio = median_by_type.iloc[0] / median_by_type.iloc[-1] if median_by_type.iloc[-1] else np.nan
        findings.append(
            f"Loại {high_type} có giá trung vị cao nhất ({median_by_type.iloc[0]:,.0f} AUD), "
            f"trong khi loại {low_type} thấp nhất ({median_by_type.iloc[-1]:,.0f} AUD); "
            f"mức chênh lệch trung vị xấp xỉ {ratio:.2f} lần."
        )

    region_stats = df.groupby("Regionname")["Price"].agg(["count", "median"])
    region_stats = region_stats[region_stats["count"] >= 30].sort_values("median", ascending=False)
    if len(region_stats) >= 2:
        findings.append(
            f"Trong các khu vực có ít nhất 30 bản ghi, {region_stats.index[0]} có giá trung vị cao nhất "
            f"({region_stats.iloc[0]['median']:,.0f} AUD); {region_stats.index[-1]} thấp nhất "
            f"({region_stats.iloc[-1]['median']:,.0f} AUD)."
        )

    corr = df["Price"].corr(df["Rooms"])
    findings.append(
        f"Hệ số tương quan Pearson giữa Price và Rooms là {corr:.2f}; trong dataset này, số phòng có mối liên hệ tuyến tính dương với giá, "
        "nhưng tương quan không chứng minh quan hệ nhân quả."
    )

    corr_distance = df["Price"].corr(df["Distance"])
    findings.append(
        f"Hệ số tương quan Pearson giữa Price và Distance là {corr_distance:.2f}; mối liên hệ tuyến tính với khoảng cách đến CBD "
        "yếu hơn quan hệ với số phòng và có thể khác nhau theo khu vực/loại bất động sản."
    )

    missing_pct = df[["BuildingArea", "YearBuilt", "CouncilArea", "Car"]].isna().mean().mul(100).sort_values(ascending=False)
    findings.append(
        f"BuildingArea có khoảng {missing_pct.get('BuildingArea', 0):.1f}% giá trị thiếu và YearBuilt có khoảng "
        f"{missing_pct.get('YearBuilt', 0):.1f}% thiếu sau bước xử lý giá trị bất hợp lý; nên thận trọng khi dùng hai biến này."
    )
    return findings


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_data()
    raw_audit = audit_data(df_raw)
    df_clean = clean_data(df_raw)
    analyses = analyze_data(df_clean)
    charts = create_charts(df_clean)
    findings = build_findings(df_clean)

    df_clean.to_csv(OUTPUT_DIR / "melb_data_clean.csv", index=False, encoding="utf-8-sig")
    raw_audit["profile"].to_csv(OUTPUT_DIR / "data_quality_profile.csv", encoding="utf-8-sig")
    for name, table in analyses.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", encoding="utf-8-sig")
    with (OUTPUT_DIR / "findings.txt").open("w", encoding="utf-8") as f:
        for i, finding in enumerate(findings, 1):
            f.write(f"{i}. {finding}\n")

    print("=" * 70)
    print("MELBOURNE HOUSING DATA PIPELINE")
    print("Raw shape:", raw_audit["shape"])
    print("Exact duplicates:", raw_audit["duplicates"])
    print("Clean shape:", df_clean.shape)
    print("\nMissing-data report (top columns):")
    print(raw_audit["profile"].head(8).to_string())
    print("\nFindings:")
    for i, finding in enumerate(findings, 1):
        print(f"{i}. {finding}")
    print("\nCharts:")
    for chart in charts:
        print(" -", chart.relative_to(PROJECT_DIR))
    print("\nAll outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
