# Kiểm thử bước làm sạch dữ liệu trong pipeline.py (không ghi tệp, không huấn luyện).
import numpy as np
import pandas as pd
import pytest

import pipeline


@pytest.fixture(scope="module")
def raw():
    return pipeline.load_data()


@pytest.fixture(scope="module")
def clean(raw):
    return pipeline.clean_data(raw)

# Tệp này không có dòng trùng và mọi dòng đều có giá hợp lệ nên không dòng nào bị xóa.
def test_clean_keeps_all_rows(raw, clean):
    assert len(raw) == 13_580
    assert len(clean) == 13_580

# Đảm bảo clean_data không sửa đổi trực tiếp DataFrame đầu vào  
def test_clean_does_not_modify_input(raw):
    before = raw.copy()
    pipeline.clean_data(raw)
    pd.testing.assert_frame_equal(raw, before)

# Đảm bảo cột Price không bị rỗng (NaN) và tất cả giá trị đều > 0
def test_price_is_always_positive(clean):
    assert clean["Price"].notna().all()
    assert (clean["Price"] > 0).all()

# Các giá trị ngoại lai (quá lớn, Bathroom = 0, YearBuilt < 1700) phải được chuyển thành NaN.
def test_outliers_become_missing(clean):
    assert not (clean["Landsize"] > pipeline.MAX_LANDSIZE).any()
    assert not (clean["BuildingArea"] > pipeline.MAX_BUILDING_AREA).any()
    assert not (clean["Bathroom"] == 0).any()
    assert not (clean["YearBuilt"] < 1700).any()

# Căn hộ không có đất riêng nên Landsize = 0 là giá trị thật, không bị xóa.
def test_landsize_zero_is_kept(clean):
    # Căn hộ không có đất riêng nên Landsize = 0 là giá trị thật, không bị xóa.
    assert (clean["Landsize"] == 0).sum() == 1_939

# Kiểm tra các cột cờ đánh dấu dữ liệu thiếu (*_was_missing) được tạo đầy đủ và chỉ mang giá trị 0 hoặc 1.
def test_missing_flags_added_and_binary(clean):
    for col in ["Car", "BuildingArea", "YearBuilt", "CouncilArea"]:
        flag = clean[f"{col}_was_missing"]
        assert set(flag.unique()) <= {0, 1}

# Các ô bị thiếu ở cột CouncilArea phải được điền bằng "Unknown" (đúng 1.369 dòng).
def test_council_area_filled_unknown(clean):
    assert clean["CouncilArea"].notna().all()
    assert (clean["CouncilArea"] == "Unknown").sum() == 1_369

# Cột Date phải ở định dạng datetime, đồng thời tách đúng cột năm (2016-2017) và tháng (1-12).
def test_date_split_into_year_and_month(clean):
    assert pd.api.types.is_datetime64_any_dtype(clean["Date"])
    assert set(clean["SaleYear"].unique()) == {2016, 2017}
    assert clean["SaleMonth"].between(1, 12).all()

# Loại bỏ các dòng trùng lặp, dòng có Price <= 0 hoặc NaN, và chuyển giá trị Rooms = 0 thành NaN.
def test_clean_drops_invalid_price_and_duplicates():
    df = pd.DataFrame({
        "Price": [500_000, 500_000, 0, -5, np.nan, 800_000],
        "Rooms": [2, 2, 3, 3, 3, 0],
    })
    out = pipeline.clean_data(df)
    assert list(out["Price"]) == [500_000, 800_000]
    assert np.isnan(out["Rooms"].iloc[1])  # Rooms = 0 chuyển thành thiếu, dòng vẫn giữ
