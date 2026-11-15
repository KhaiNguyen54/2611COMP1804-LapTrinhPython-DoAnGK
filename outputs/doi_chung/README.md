# Đối chứng: mô hình trước khi loại ngoại lai

Hai tệp CSV trong thư mục này là kết quả của `train_model.py` khi dữ liệu **chưa** áp dụng ba quy tắc ngoại lai
(`Landsize > 10000`, `BuildingArea > 1000`, `Bathroom == 0`). Cách tạo, trên một bản sao của dự án:

1. Trong `pipeline.py` đặt `MAX_LANDSIZE = float("inf")`, `MAX_BUILDING_AREA = float("inf")`
   và thay dòng `df.loc[df["Bathroom"] == 0, "Bathroom"] = np.nan` bằng `pass`.
2. Chạy `python pipeline.py` rồi `python train_model.py`.
3. Lấy `outputs/model_metrics.csv` và `outputs/model_validation_metrics.csv`.

So sánh với kết quả sau khi loại ngoại lai: `outputs/model_metrics.csv` và `outputs/model_validation_metrics.csv` của dự án.
Số đo có thể lệch nhẹ giữa các máy do phiên bản thư viện (Ridge trên validation lệch khoảng 23 AUD giữa hai máy đã thử).
