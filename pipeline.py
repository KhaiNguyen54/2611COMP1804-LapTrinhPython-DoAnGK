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

def build_report_metrics(df_raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Các số liệu bổ sung mà báo cáo trích (nhóm N, L, P). Mỗi dòng có mã, nội dung, giá trị và cách tính.

    Ghi ra outputs/report_metrics.csv; báo cáo (report/) dẫn các mã này ở dòng "Nguồn".
    Chỉ đọc dữ liệu, không huấn luyện mô hình.
    """
    rows = []

    def r(ma, noi_dung, gia_tri, cach_tinh):
        if isinstance(gia_tri, (float, np.floating)):
            gia_tri = round(float(gia_tri), 4)
        rows.append({"ma": ma, "noi_dung": noi_dung, "gia_tri": gia_tri, "cach_tinh": cach_tinh})

    raw = df_raw
    date = pd.to_datetime(raw["Date"], dayfirst=True)
    price = clean["Price"]
    RAW = "data/melb_data.csv"
    CLN = "outputs/melb_data_clean.csv"

    # ---------------------------------------------------------------- N: nguồn dữ liệu (tệp gốc)
    r("N01", "Số dòng / số cột tệp gốc", f"{raw.shape[0]} / {raw.shape[1]}", f"{RAW}: shape")
    r("N02", "Ngày bán đầu tiên / cuối cùng", f"{date.min():%d/%m/%Y} / {date.max():%d/%m/%Y}", "pd.to_datetime(Date, dayfirst=True).min()/max()")
    for c in ["Suburb", "Postcode", "Regionname", "SellerG", "Address", "CouncilArea"]:
        r("N03", f"Số giá trị khác nhau của {c} (không tính thiếu)", int(raw[c].nunique()), f"{RAW}: {c}.nunique()")
    r("N04", "Số dòng có ít nhất một ô null", int(raw.isna().any(axis=1).sum()), f"{RAW}: isna().any(axis=1).sum()")
    r("N05", "Tổng số ô null", int(raw.isna().sum().sum()), f"{RAW}: isna().sum().sum()")
    r("N06", "Tỷ lệ dòng có Bedroom2 bằng Rooms (%)", (raw["Bedroom2"] == raw["Rooms"]).mean() * 100, f"{RAW}: (Bedroom2 == Rooms).mean()")
    num_cols = raw.select_dtypes("number").columns
    r("N07", "Số cột chuỗi / số nguyên / số thực",
      f"{raw.shape[1] - len(num_cols)} / {sum(pd.api.types.is_integer_dtype(t) for t in raw.dtypes)} / {sum(pd.api.types.is_float_dtype(t) for t in raw.dtypes)}",
      f"{RAW}: dtypes")
    r("N08", "Số dòng trùng hoàn toàn", int(raw.duplicated().sum()), f"{RAW}: duplicated().sum()")
    tc = raw["Type"].value_counts()
    r("N09", "Cơ cấu loại nhà h / t / u (số dòng, %)",
      "; ".join(f"{k}={tc[k]} ({tc[k] / len(raw) * 100:.1f}%)" for k in ["h", "t", "u"]), f"{RAW}: Type.value_counts()")
    rg = raw["Regionname"].value_counts()
    metro = ["Southern Metropolitan", "Northern Metropolitan", "Western Metropolitan", "Eastern Metropolitan"]
    r("N10", "Tỷ lệ bốn vùng Metropolitan lớn (%)", "; ".join(f"{k}={rg[k] / len(raw) * 100:.1f}" for k in metro), f"{RAW}: Regionname.value_counts()")
    r("N10", "Tổng bốn vùng Metropolitan lớn (%)", rg[metro].sum() / len(raw) * 100, f"{RAW}: Regionname.value_counts()")
    vic = [k for k in rg.index if "Victoria" in k]
    r("N10", "Ba vùng Victoria: số dòng / %", f"{int(rg[vic].sum())} / {rg[vic].sum() / len(raw) * 100:.2f}", f"{RAW}: Regionname.value_counts()")
    ym = date.dt.to_period("M").value_counts().sort_index()
    r("N11", "Số tháng có giao dịch", len(ym), "to_period('M').nunique()")
    r("N11", "Giao dịch tháng 1/2016; tháng 2/2016; quý 1/2016", f"{ym['2016-01']}; {ym['2016-02']}; {ym['2016-01'] + ym['2016-02']}", "to_period('M').value_counts()")
    r("N11", "Min / max số giao dịch các tháng còn lại", f"{ym.drop([pd.Period('2016-01'), pd.Period('2016-02')]).min()} / {ym.max()}", "to_period('M').value_counts()")
    r("N11", "Tháng không có giao dịch", ", ".join(str(m) for m in pd.period_range(ym.index.min(), ym.index.max(), freq="M").difference(ym.index)), "period_range trừ các tháng có giao dịch")
    r("N12", "Price nhỏ nhất / lớn nhất / trung vị / trung bình",
      f"{raw.Price.min():.0f} / {raw.Price.max():.0f} / {raw.Price.median():.0f} / {raw.Price.mean():.0f}", f"{RAW}: Price")

    # ---------------------------------------------------------------- L: làm sạch
    sold_year = date.dt.year
    inv = {
        "Price <= 0": (raw.Price <= 0).sum(), "Rooms <= 0": (raw.Rooms <= 0).sum(),
        "Distance < 0": (raw.Distance < 0).sum(), "Landsize < 0": (raw.Landsize < 0).sum(),
        "BuildingArea <= 0": (raw.BuildingArea <= 0).sum(), "YearBuilt < 1700": (raw.YearBuilt < 1700).sum(),
        "Landsize == 0": (raw.Landsize == 0).sum(), f"Landsize > {MAX_LANDSIZE}": (raw.Landsize > MAX_LANDSIZE).sum(),
        f"BuildingArea > {MAX_BUILDING_AREA}": (raw.BuildingArea > MAX_BUILDING_AREA).sum(),
        "Bathroom == 0": (raw.Bathroom == 0).sum(), "YearBuilt >= 2017": (raw.YearBuilt >= 2017).sum(),
        "YearBuilt > năm bán": (raw.YearBuilt > sold_year).sum(), "Distance == 0": (raw.Distance == 0).sum(),
        "Rooms >= 7": (raw.Rooms >= 7).sum(),
        "Tọa độ ngoài khung Melbourne": (~raw.Lattitude.between(-39.5, -36.0) | ~raw.Longtitude.between(142.0, 147.0)).sum(),
    }
    for k, v in inv.items():
        r("L01", f"Số dòng thỏa điều kiện {k} (tệp gốc)", int(v), f"{RAW}: {k}")
    r("L01", "YearBuilt nhỏ nhất", raw.YearBuilt.min(), f"{RAW}: YearBuilt.min()")
    r("L01", "Lattitude min / max", f"{raw.Lattitude.min():.2f} / {raw.Lattitude.max():.2f}", f"{RAW}: Lattitude.min()/max()")
    r("L01", "Longtitude min / max", f"{raw.Longtitude.min():.2f} / {raw.Longtitude.max():.2f}", f"{RAW}: Longtitude.min()/max()")
    u = raw.Type == "u"
    r("L02", "Landsize == 0 và là căn hộ: số dòng / % tổng số căn hộ", f"{int(((raw.Landsize == 0) & u).sum())} / {((raw.Landsize == 0) & u).sum() / u.sum() * 100:.0f}", f"{RAW}: (Landsize == 0) & (Type == 'u')")
    r("L02", "Tỷ lệ căn hộ trong các dòng Landsize == 0 (%)", ((raw.Landsize == 0) & u).sum() / (raw.Landsize == 0).sum() * 100, f"{RAW}")
    big = raw[raw.Landsize > MAX_LANDSIZE]
    r("L02", "Landsize > 10000: lớn nhất / số căn hộ", f"{big.Landsize.max():.0f} / {int((big.Type == 'u').sum())}", f"{RAW}")
    r("L02", "BuildingArea > 1000: lớn nhất", raw.loc[raw.BuildingArea > MAX_BUILDING_AREA, "BuildingArea"].max(), f"{RAW}")
    for c in ["BuildingArea", "YearBuilt", "Landsize", "Bathroom", "Car", "CouncilArea"]:
        n = int(clean[c].isna().sum())
        r("L03", f"Số ô thiếu của {c} sau làm sạch (%)", f"{n} ({n / len(clean) * 100:.1f}%)", f"{CLN}: {c}.isna().sum()")
    both = clean[["BuildingArea", "YearBuilt"]].isna().any(axis=1)
    r("L03", "Số dòng thiếu BuildingArea hoặc YearBuilt (%)", f"{int(both.sum())} ({both.mean() * 100:.1f}%)", f"{CLN}: isna().any(axis=1)")
    q1, q3 = price.quantile(.25), price.quantile(.75)
    ub = q3 + 1.5 * (q3 - q1)
    r("L04", "Q1 / Q3 / ngưỡng trên của giá", f"{q1:.0f} / {q3:.0f} / {ub:.0f}", f"{CLN}: Q3 + 1.5 * IQR")
    r("L04", "Số dòng vượt ngưỡng trên (%)", f"{int((price > ub).sum())} ({(price > ub).mean() * 100:.1f}%)", f"{CLN}: Price > ngưỡng")
    r("L04", "Độ lệch của giá / của log(giá)", f"{price.skew():.2f} / {np.log(price).skew():.2f}", f"{CLN}: Price.skew(), np.log(Price).skew()")
    mg = clean.loc[clean.Price.idxmax()]
    r("L04", "Giao dịch giá cao nhất: suburb, giá, số phòng, khoảng cách", f"{mg.Suburb}, {mg.Price:.0f}, {mg.Rooms:.0f}, {mg.Distance}", f"{CLN}: Price.idxmax()")
    for c in ["BuildingArea_was_missing", "YearBuilt_was_missing"]:
        g = clean.groupby(c).Price.median()
        r("L05", f"Giá trung vị khi {c} = 1 / 0", f"{g[1]:.0f} / {g[0]:.0f}", f"{CLN}: groupby({c}).Price.median()")
    r("L05", "Giá trung vị khi Landsize == 0 / Landsize > 0", f"{price[clean.Landsize == 0].median():.0f} / {price[clean.Landsize > 0].median():.0f}", f"{CLN}")

    # ---------------------------------------------------------------- P: phân tích
    q = price.quantile([.1, .9, .99])
    r("P01", "Độ lệch chuẩn / hệ số biến thiên của giá", f"{price.std():.0f} / {price.std() / price.mean():.2f}", f"{CLN}: Price.std()/Price.mean()")
    r("P01", "Phân vị 10% / 90% / 99%", f"{q[.1]:.0f} / {q[.9]:.0f} / {q[.99]:.0f}", f"{CLN}: Price.quantile([.1, .9, .99])")
    r("P01", "Trung bình cao hơn trung vị (%)", (price.mean() / price.median() - 1) * 100, f"{CLN}")
    r("P01", "Phân vị 90% / phân vị 10%", q[.9] / q[.1], f"{CLN}")
    for c in ["Rooms", "Bathroom", "Car", "Distance", "Landsize", "BuildingArea", "YearBuilt"]:
        s = clean[c]
        r("P02", f"{c}: số dòng có giá trị; trung bình; trung vị; lớn nhất", f"{s.count()}; {s.mean():.1f}; {s.median():g}; {s.max():g}", f"{CLN}: {c}.describe()")
    old = raw.assign(BuildingArea=raw.BuildingArea.where(raw.BuildingArea > 0))  # tệp trước khi áp dụng ba quy tắc ngoại lai
    r("P02", "Landsize trung bình / lớn nhất trước khi loại ngoại lai", f"{old.Landsize.mean():.1f} / {old.Landsize.max():.0f}", f"{RAW} (BuildingArea <= 0 -> NaN)")
    r("P02", "BuildingArea trung bình / lớn nhất trước khi loại ngoại lai", f"{old.BuildingArea.mean():.1f} / {old.BuildingArea.max():.0f}", f"{RAW} (BuildingArea <= 0 -> NaN)")
    g = clean.groupby("Type").Price
    r("P03", "Hệ số biến thiên theo loại nhà h / t / u", " / ".join(f"{(g.std() / g.mean())[t]:.2f}" for t in "htu"), f"{CLN}: groupby(Type).Price.std()/mean()")
    r("P03", "Trung vị giá nhà riêng / căn hộ", price[clean.Type == "h"].median() / price[clean.Type == "u"].median(), f"{CLN}")
    for c in ["BuildingArea", "Rooms", "Distance", "YearBuilt"]:
        m = clean.groupby("Type")[c].median()
        r("P03", f"Trung vị {c} theo loại nhà h / t / u", " / ".join(f"{m[t]:g}" for t in "htu"), f"{CLN}: groupby(Type).{c}.median()")
    rc = clean.groupby("Regionname").Price.median()
    r("P04", "Trung vị Southern / Western Metropolitan", rc["Southern Metropolitan"] / rc["Western Metropolitan"], f"{CLN}: groupby(Regionname).Price.median()")
    rm = clean.groupby("Rooms").Price.median()
    r("P05", "Tăng trung vị theo số phòng 2..6 (%)", ", ".join(f"{k:g}:{v:+.0f}" for k, v in (rm.pct_change() * 100).loc[2:6].items()), f"{CLN}: groupby(Rooms).Price.median().pct_change()")
    r("P05", "Trung vị 4 phòng / 1 phòng", rm[4] / rm[1], f"{CLN}")
    vc = clean.Rooms.value_counts()
    r("P05", "Nhóm số phòng dưới 20 giao dịch: số nhóm; tổng số dòng", f"{int((vc < 20).sum())}; {int(vc[vc < 20].sum())}", f"{CLN}: Rooms.value_counts()")
    sc = clean.groupby("Suburb").Price.agg(["count", "median"])
    r("P06", "Số suburb; suburb từ 30 giao dịch; suburb dưới 5 giao dịch", f"{len(sc)}; {int((sc['count'] >= 30).sum())}; {int((sc['count'] < 5).sum())}", f"{CLN}: groupby(Suburb)")
    ts = sc[sc["count"] >= 30].sort_values("median", ascending=False).head(10)  # như analyze_data()["top_suburbs"]
    r("P06", "Trung vị suburb đứng đầu / trung vị chung; suburb thứ 10 / trung vị chung", f"{ts['median'].iloc[0] / price.median():.2f}; {ts['median'].iloc[-1] / price.median():.2f}", "outputs/top_suburbs.csv (cùng cách tính với analyze_data)")
    cols = ["Price", "Rooms", "Bedroom2", "Bathroom", "Car", "Distance", "Landsize", "BuildingArea", "YearBuilt", "Propertycount"]
    pe = clean[cols].corr()["Price"]
    sp = clean[cols].corr(method="spearman")["Price"]
    for c in cols[1:]:
        r("P07", f"Pearson / Spearman của {c} với Price", f"{pe[c]:.3f} / {sp[c]:.3f}", f"{CLN}: df[cols].corr(), corr(method='spearman')")
    r("P07", "Pearson trước khi loại ngoại lai: BuildingArea; Landsize (với Price)", f"{old.Price.corr(old.BuildingArea):.3f}; {old.Price.corr(old.Landsize):.3f}", f"{RAW} (BuildingArea <= 0 -> NaN)")
    lp = clean[clean.Landsize > 0]
    r("P07", "Landsize > 0: số dòng; Pearson; Spearman với Price", f"{len(lp)}; {lp.Price.corr(lp.Landsize):.3f}; {lp.Price.corr(lp.Landsize, method='spearman'):.3f}", f"{CLN}: Landsize > 0")
    for c in ["Distance", "Landsize", "YearBuilt"]:
        out = []
        for t in "htu":
            e_ = clean[(clean.Type == t) & ((clean.Landsize > 0) if c == "Landsize" else True)]
            out.append(f"{t}={e_.Price.corr(e_[c]):.3f}")
        r("P07", f"Pearson của {c} với Price theo loại nhà" + (" (chỉ Landsize > 0)" if c == "Landsize" else ""), "; ".join(out), f"{CLN}: Type == t")
    r("P07", "Số dòng có YearBuilt", int(clean.YearBuilt.notna().sum()), f"{CLN}")
    r("P07", "Pearson giữa thuộc tính: Rooms-Bedroom2; Rooms-Bathroom; Landsize-BuildingArea; YearBuilt-Distance",
      f"{clean.Rooms.corr(clean.Bedroom2):.3f}; {clean.Rooms.corr(clean.Bathroom):.3f}; {clean.Landsize.corr(clean.BuildingArea):.3f}; {clean.YearBuilt.corr(clean.Distance):.3f}", f"{CLN}")
    mo = clean.groupby("SaleMonth")["Price"].median()  # như analyze_data()["price_by_sale_month"]
    r("P08", "Trung vị tháng 7 so với tháng 6; tháng 8 (%)", f"{(mo[7] / mo[6] - 1) * 100:.1f}; {(mo[7] / mo[8] - 1) * 100:.1f}", "outputs/price_by_sale_month.csv (cùng cách tính với analyze_data)")
    for yr in (2016, 2017):
        e_ = clean[clean.SaleYear == yr]
        r("P08", f"Năm {yr}: số giao dịch; giá trung vị", f"{len(e_)}; {e_.Price.median():.0f}", f"{CLN}: SaleYear == {yr}")
    r("P08", "Trung vị 2017 so với 2016 (%)", (clean[clean.SaleYear == 2017].Price.median() / clean[clean.SaleYear == 2016].Price.median() - 1) * 100, f"{CLN}")
    for mth in (6, 7, 8):
        e_ = clean[(clean.SaleYear == 2017) & (clean.SaleMonth == mth)]
        r("P08", f"Tháng {mth}/2017: số giao dịch; trung vị; % Southern Metropolitan", f"{len(e_)}; {e_.Price.median():.0f}; {(e_.Regionname == 'Southern Metropolitan').mean() * 100:.1f}", f"{CLN}")
    r("P08", "% Southern Metropolitan toàn bộ dữ liệu", (clean.Regionname == "Southern Metropolitan").mean() * 100, f"{CLN}")
    h, edges = np.histogram(price, bins=45)
    r("P09", "Dải của cột cao nhất trong histogram 45 khoảng (AUD)", f"{edges[h.argmax()]:.0f} - {edges[h.argmax() + 1]:.0f}", f"{CLN}: np.histogram(Price, bins=45)")
    r("P09", "Tỷ lệ giao dịch dưới 2 triệu (%)", (price < 2e6).mean() * 100, f"{CLN}")
    r("P09", "Giao dịch trên 2 triệu: số dòng; % nằm trong 15 km", f"{int((price > 2e6).sum())}; {(clean[price > 2e6].Distance <= 15).mean() * 100:.1f}", f"{CLN}")
    r("P09", "Giao dịch ngoài 20 km: số dòng; % dưới 1.5 triệu", f"{int((clean.Distance > 20).sum())}; {(clean[clean.Distance > 20].Price < 1.5e6).mean() * 100:.1f}", f"{CLN}")
    r("P09", "Giá trung vị theo Method S; SP", f"{price[clean.Method == 'S'].median():.0f}; {price[clean.Method == 'SP'].median():.0f}", f"{CLN}: groupby(Method)")

    return pd.DataFrame(rows)


def create_quality_charts(df_raw: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Hai biểu đồ chất lượng dữ liệu vẽ từ tệp gốc: dữ liệu thiếu theo cột và số dòng bị ảnh hưởng theo quy tắc."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 8. Tỷ lệ dữ liệu thiếu theo cột (tệp gốc).
    missing = (df_raw.isna().mean() * 100)
    missing = missing[missing > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    bars = ax.barh(missing.index, missing.values, color="#4f81bd")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlim(0, 58)
    ax.set(title="Dữ liệu thiếu trong tệp gốc", xlabel=f"Tỷ lệ dữ liệu thiếu (% trên {len(df_raw):,} bản ghi)", ylabel="Cột")
    fig.tight_layout()
    path = output_dir / "08_missing_by_column.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    saved.append(path)

    # 9. Số dòng bị ảnh hưởng theo từng quy tắc của clean_data().
    rules = {
        "CouncilArea thiếu → Unknown": int(df_raw["CouncilArea"].isna().sum()),
        "BuildingArea ≤ 0 → thiếu": int((df_raw["BuildingArea"] <= 0).sum()),
        "YearBuilt < 1700 → thiếu": int((df_raw["YearBuilt"] < 1700).sum()),
        "Bathroom = 0 → thiếu": int((df_raw["Bathroom"] == 0).sum()),
        f"Landsize > {MAX_LANDSIZE:,} → thiếu": int((df_raw["Landsize"] > MAX_LANDSIZE).sum()),
        f"BuildingArea > {MAX_BUILDING_AREA:,} → thiếu": int((df_raw["BuildingArea"] > MAX_BUILDING_AREA).sum()),
        "Price ≤ 0 (bỏ dòng)": int((df_raw["Price"] <= 0).sum()),
        "Rooms ≤ 0 → thiếu": int((df_raw["Rooms"] <= 0).sum()),
        "Distance < 0 → thiếu": int((df_raw["Distance"] < 0).sum()),
        "Landsize < 0 → thiếu": int((df_raw["Landsize"] < 0).sum()),
        "Tọa độ ngoài vùng → thiếu": int(
            (~df_raw["Lattitude"].between(-39.5, -36.0)).sum() + (~df_raw["Longtitude"].between(142.0, 147.0)).sum()
        ),
        "Dòng trùng hoàn toàn (bỏ dòng)": int(df_raw.duplicated().sum()),
    }
    effect = pd.Series(rules).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    bars = ax.barh(effect.index, effect.values, color="#c0504d")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlim(0, 1600)
    ax.set(title="Số dòng bị ảnh hưởng theo từng quy tắc", xlabel="Số dòng bị ảnh hưởng", ylabel="Quy tắc làm sạch")
    fig.tight_layout()
    path = output_dir / "09_cleaning_rule_effect.png"
    fig.savefig(path, dpi=200)
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
    charts = create_charts(df_clean) + create_quality_charts(df_raw)
    report_metrics = build_report_metrics(df_raw, df_clean)
    findings = build_findings(df_clean)

    df_clean.to_csv(OUTPUT_DIR / "melb_data_clean.csv", index=False, encoding="utf-8-sig")
    raw_audit["profile"].to_csv(OUTPUT_DIR / "data_quality_profile.csv", encoding="utf-8-sig")
    for name, table in analyses.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", encoding="utf-8-sig")
    report_metrics.to_csv(OUTPUT_DIR / "report_metrics.csv", index=False, encoding="utf-8-sig")
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
