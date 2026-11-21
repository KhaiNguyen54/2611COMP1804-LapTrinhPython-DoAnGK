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


def test_clean_keeps_all_rows(raw, clean):
    # Tệp này không có dòng trùng và mọi dòng đều có giá hợp lệ nên không dòng nào bị xóa.
    assert len(raw) == 13_580
    assert len(clean) == 13_580


def test_clean_does_not_modify_input(raw):
    before = raw.copy()
    pipeline.clean_data(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_price_is_always_positive(clean):
    assert clean["Price"].notna().all()
    assert (clean["Price"] > 0).all()


def test_outliers_become_missing(clean):
    assert not (clean["Landsize"] > pipeline.MAX_LANDSIZE).any()
    assert not (clean["BuildingArea"] > pipeline.MAX_BUILDING_AREA).any()
    assert not (clean["Bathroom"] == 0).any()
    assert not (clean["YearBuilt"] < 1700).any()


def test_landsize_zero_is_kept(clean):
    # Căn hộ không có đất riêng nên Landsize = 0 là giá trị thật, không bị xóa.
    assert (clean["Landsize"] == 0).sum() == 1_939


def test_missing_flags_added_and_binary(clean):
    for col in ["Car", "BuildingArea", "YearBuilt", "CouncilArea"]:
        flag = clean[f"{col}_was_missing"]
        assert set(flag.unique()) <= {0, 1}


def test_council_area_filled_unknown(clean):
    assert clean["CouncilArea"].notna().all()
    assert (clean["CouncilArea"] == "Unknown").sum() == 1_369


def test_date_split_into_year_and_month(clean):
    assert pd.api.types.is_datetime64_any_dtype(clean["Date"])
    assert set(clean["SaleYear"].unique()) == {2016, 2017}
    assert clean["SaleMonth"].between(1, 12).all()


def test_clean_drops_invalid_price_and_duplicates():
    df = pd.DataFrame({
        "Price": [500_000, 500_000, 0, -5, np.nan, 800_000],
        "Rooms": [2, 2, 3, 3, 3, 0],
    })
    out = pipeline.clean_data(df)
    assert list(out["Price"]) == [500_000, 800_000]
    assert np.isnan(out["Rooms"].iloc[1])  # Rooms = 0 chuyển thành thiếu, dòng vẫn giữ
