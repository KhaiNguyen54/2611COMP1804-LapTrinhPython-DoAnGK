"""Tính các số liệu trong báo cáo mà không có sẵn trong outputs/.

Chạy từ thư mục gốc của dự án, sau khi đã chạy pipeline.py và train_model.py:

    python report/so_lieu_bo_sung.py

Kết quả ghi vào report/so_lieu_bo_sung.csv với bốn cột:
  ma         mã số liệu (N: nguồn dữ liệu, L: làm sạch, P: phân tích, M: mô hình)
  noi_dung   số liệu đo cái gì
  gia_tri    giá trị tính được
  cach_tinh  biểu thức hoặc tệp nguồn
Báo cáo trích các mã này ở phần "Nguồn" dưới mỗi bảng và hình (Phụ lục B).
"""
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline as P  # noqa: E402

OUT = ROOT / "outputs"
rows = []


def r(ma, noi_dung, gia_tri, cach_tinh):
    if isinstance(gia_tri, (float, np.floating)):
        gia_tri = round(float(gia_tri), 4)
    rows.append({"ma": ma, "noi_dung": noi_dung, "gia_tri": gia_tri, "cach_tinh": cach_tinh})


raw = pd.read_csv(ROOT / "data" / "melb_data.csv")
clean = pd.read_csv(OUT / "melb_data_clean.csv", parse_dates=["Date"])
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
    "Landsize == 0": (raw.Landsize == 0).sum(), f"Landsize > {P.MAX_LANDSIZE}": (raw.Landsize > P.MAX_LANDSIZE).sum(),
    f"BuildingArea > {P.MAX_BUILDING_AREA}": (raw.BuildingArea > P.MAX_BUILDING_AREA).sum(),
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
big = raw[raw.Landsize > P.MAX_LANDSIZE]
r("L02", "Landsize > 10000: lớn nhất / số căn hộ", f"{big.Landsize.max():.0f} / {int((big.Type == 'u').sum())}", f"{RAW}")
r("L02", "BuildingArea > 1000: lớn nhất", raw.loc[raw.BuildingArea > P.MAX_BUILDING_AREA, "BuildingArea"].max(), f"{RAW}")
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

# chia tập đúng như train_model.py (hai lần train_test_split, random_state=42)
y = clean.loc[clean.Price.notna() & (clean.Price > 0), "Price"].astype(float)
tr_full, te = train_test_split(y, test_size=0.2, random_state=42)
tr, va = train_test_split(tr_full, test_size=0.2, random_state=42)
for ten, s in [("train", tr), ("validation", va), ("test", te)]:
    r("L06", f"Tập {ten}: số dòng / tỷ lệ / giá trung vị", f"{len(s)} / {len(s) / len(y) * 100:.0f}% / {s.median():.0f}", "train_test_split(test_size=0.2, random_state=42) hai lần")
r("L06", "Train gộp validation (dùng huấn luyện lại)", len(tr_full), "train_test_split lần 1")

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
ts = pd.read_csv(OUT / "top_suburbs.csv")
r("P06", "Trung vị suburb đứng đầu / trung vị chung; suburb thứ 10 / trung vị chung", f"{ts['median'].iloc[0] / price.median():.2f}; {ts['median'].iloc[-1] / price.median():.2f}", "outputs/top_suburbs.csv")
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
mo = pd.read_csv(OUT / "price_by_sale_month.csv").set_index("SaleMonth")["median"]
r("P08", "Trung vị tháng 7 so với tháng 6; tháng 8 (%)", f"{(mo[7] / mo[6] - 1) * 100:.1f}; {(mo[7] / mo[8] - 1) * 100:.1f}", "outputs/price_by_sale_month.csv")
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

# ---------------------------------------------------------------- M: mô hình (từ các tệp kết quả của train_model.py)
vm = pd.read_csv(OUT / "model_validation_metrics.csv").set_index("Model")
tm = pd.read_csv(OUT / "model_metrics.csv").set_index("Model")
r("M01", "MAE validation của Random Forest thấp hơn Ridge (%)", (1 - vm.loc["RandomForest_log_target", "MAE_AUD"] / vm.loc["Ridge", "MAE_AUD"]) * 100, "outputs/model_validation_metrics.csv")
r("M01", "MAE test của Random Forest thấp hơn baseline (%)", (1 - tm.loc["RandomForest_log_target", "MAE_AUD"] / tm.loc["Baseline_median", "MAE_AUD"]) * 100, "outputs/model_metrics.csv")
t = pd.read_csv(OUT / "test_predictions.csv")
a, p = t.ActualPrice, t.PredictedPrice
e = a - p
ape = e.abs() / a
r("M02", "Số dòng test; hệ số tương quan giữa giá thực và giá dự đoán", f"{len(t)}; {np.corrcoef(a, p)[0, 1]:.3f}", "outputs/test_predictions.csv")
r("M02", "MAPE; sai số tương đối trung vị (%)", f"{ape.mean() * 100:.1f}; {ape.median() * 100:.1f}", "|thực - dự đoán| / thực")
r("M02", "Tỷ lệ lệch không quá 10%; 20% (%)", f"{(ape <= .1).mean() * 100:.1f}; {(ape <= .2).mean() * 100:.1f}", "|thực - dự đoán| / thực")
r("M02", "Sai số (thực - dự đoán): trung vị; trung bình", f"{e.median():.0f}; {e.mean():.0f}", "outputs/test_predictions.csv: Residual")
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

res = pd.DataFrame(rows)
res.to_csv(ROOT / "report" / "so_lieu_bo_sung.csv", index=False, encoding="utf-8-sig")
print(f"đã ghi {len(res)} số liệu vào report/so_lieu_bo_sung.csv")
