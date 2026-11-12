# Melbourne Housing — Data Analysis Pipeline

Project nền cho bài phân tích dữ liệu Python dựa trên `melb_data.csv`.

## 1. Dataset và nguồn

- Dataset: **Melbourne Housing Snapshot**.
- Kích thước file đầu vào: 13.580 dòng × 21 cột.
- Theo trang dataset trên Kaggle, dữ liệu là snapshot của bộ dữ liệu Melbourne Housing do Tony Pino tạo; dữ liệu được thu thập từ các tin đăng công khai trên Domain.com.au.
- Nguồn: https://www.kaggle.com/dansbecker/melbourne-housing-snapshot
- Giấy phép được nêu tại Kaggle: CC BY-NC-SA 4.0. Giữ thông tin ghi công và tuân thủ giấy phép nếu chia sẻ lại hoặc biến đổi dữ liệu.

## 2. Cấu trúc thư mục

```text
melbourne_pipeline_project/
├── data/
│   └── melb_data.csv
├── outputs/                 # tự tạo kết quả khi chạy
├── pipeline.py              # pipeline Python có thể chạy trực tiếp
├── melbourne_data_analysis_pipeline.ipynb  # notebook triển khai theo từng bước
└── README.md
```

## 3. Cài đặt môi trường

Tạo môi trường ảo (tùy chọn nhưng nên dùng):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy matplotlib seaborn jupyter
```

## 4. Chạy pipeline Python

Từ thư mục dự án:

```powershell
python pipeline.py
```

Kết quả được lưu trong `outputs/`: dữ liệu sau làm sạch, báo cáo chất lượng, bảng phân tích, nhận xét và 5 biểu đồ.

## 5. Chạy notebook

Mở `melbourne_data_analysis_pipeline.ipynb` trong VS Code/Jupyter và chạy lần lượt các cell từ trên xuống. Notebook sử dụng đường dẫn `data/melb_data.csv` tương đối với thư mục dự án.

## 6. Quy tắc xử lý cần trình bày trong báo cáo

- Dữ liệu gốc không bị ghi đè.
- Dòng trùng hoàn toàn được loại bỏ.
- `Date` chuyển sang kiểu datetime với `dayfirst=True` theo định dạng ngày/tháng/năm của bộ dữ liệu.
- Giá bán không dương, số phòng không dương, khoảng cách/diện tích đất âm và diện tích xây dựng không dương được đánh dấu thiếu; `YearBuilt < 1700` được xem là không hợp lý và chuyển thành thiếu.
- `BuildingArea` và `YearBuilt` có tỷ lệ thiếu lớn; mặc định không nội suy tuỳ tiện trong pipeline EDA. Các cột này vẫn còn NaN để tránh tạo ra số liệu giả cho thống kê mô tả.
- Thêm cờ `*_was_missing` cho một số cột quan trọng trước bước sửa giá trị không hợp lý. Giá trị bị chuyển thành thiếu do kiểm tra tính hợp lệ không nhất thiết là giá trị thiếu ban đầu; nếu dùng cho mô hình ML, cần tách riêng hai loại cờ.
- Giá ngoại lệ không tự động xóa: dùng biểu đồ/ thống kê để nhận diện và nêu lý do trước khi quyết định.

## 7. Phạm vi phân tích trong phiên bản nền

- Kiểm tra schema, kiểu dữ liệu, số lượng thiếu, dòng trùng và giá trị không hợp lý.
- Thống kê mô tả giá bán.
- So sánh giá theo loại bất động sản, vùng, số phòng và suburb.
- Phân tích tương quan giá với khoảng cách, số phòng và một số thuộc tính số.
- 5 biểu đồ: histogram, boxplot, bar chart, scatter plot và correlation heatmap.
- Sinh ít nhất 3 nhận xét dựa trên kết quả tính toán thực tế.

Lưu ý: đây là bộ dữ liệu snapshot giai đoạn 2016–2017 và thuộc thị trường Melbourne, Australia; không đại diện cho giá nhà hiện tại hoặc thị trường Việt Nam.

## 8. Xây dựng mô hình Regression

Sau khi chạy `pipeline.py` để tạo `outputs/melb_data_clean.csv`, chạy:

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

`train_model.py` thực hiện các bước:

1. Chọn `Price` làm biến mục tiêu; dùng các thuộc tính nhà ở và vị trí làm biến đầu vào.
2. Chia dữ liệu thành train (80%) và test (20%), với `random_state=42`.
3. So sánh baseline dự đoán giá trung vị, Ridge Regression và Random Forest.
4. Đặt bước điền giá trị thiếu và One-Hot Encoding bên trong `Pipeline` để chỉ học quy tắc tiền xử lý từ tập train.
5. Huấn luyện trên `log1p(Price)` vì giá có phân phối lệch phải; dự đoán được chuyển ngược về AUD trước khi tính metric.
6. So sánh bằng MAE, RMSE và R²; chọn mô hình có MAE thấp nhất trên tập test.

Kết quả được ghi vào:

- `models/best_pipeline.joblib`: mô hình tốt nhất đã fit trong môi trường Python hiện tại.
- `models/model_metadata.json`: tên mô hình, các thuộc tính và cấu hình đánh giá.
- `outputs/model_metrics.csv`: bảng so sánh mô hình.
- `outputs/test_predictions.csv`: giá thực tế, giá dự đoán và residual của tập test.
- `outputs/06_actual_vs_predicted.png`, `outputs/07_residual_distribution.png`: biểu đồ chẩn đoán.

**Lưu ý đánh giá:** bộ dữ liệu là snapshot lịch sử Melbourne, không phải dữ liệu giá hiện tại. Chia ngẫu nhiên phù hợp để thiết lập baseline cho bài tập; nếu mục tiêu là dự đoán cho giai đoạn tương lai, nên dùng chia tập theo thời gian. Hãy chạy `train_model.py` trong đúng môi trường của ứng dụng để tự sinh file `.joblib`; không nên chuyển file mô hình đã lưu giữa các phiên bản thư viện khác nhau mà chưa kiểm tra tương thích.
