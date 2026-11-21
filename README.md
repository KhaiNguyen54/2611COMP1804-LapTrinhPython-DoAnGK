# Melbourne Housing — Data Analysis Pipeline

Đồ án giữa kỳ môn Lập trình Python: phân tích dữ liệu giá nhà Melbourne từ `melb_data.csv` và xây dựng mô hình hồi quy dự đoán giá bằng scikit-learn.

## 1. Dataset và nguồn

- Dataset: **Melbourne Housing Snapshot**.
- Kích thước file đầu vào: 13.580 dòng × 21 cột.
- Theo trang dataset trên Kaggle, dữ liệu là snapshot của bộ dữ liệu Melbourne Housing do Tony Pino tạo, thu thập bằng phương pháp cào dữ liệu web (web scraping) từ các tin đăng công khai qua https://www.domain.com.au. Dự án này không tự cào dữ liệu: tệp CSV được tải thủ công từ Kaggle và đặt vào `data/`.
- Thông tin bộ dữ liệu: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
- Giấy phép: CC BY-NC-SA 4.0. Giữ thông tin ghi công và tuân thủ giấy phép nếu chia sẻ lại hoặc biến đổi dữ liệu.

## 2. Cấu trúc thư mục

```text
2611COMP1804-LapTrinhPython-DoAnGK/
├── data/
│   └── melb_data.csv        # dữ liệu gốc, không sửa tay
├── outputs/                 # kết quả tạo ra khi chạy pipeline.py và train_model.py
│   └── doi_chung/           # kết quả mô hình trước khi loại ngoại lai, để đối chiếu (có README riêng)
├── models/                  # tự tạo khi chạy train_model.py (không đưa lên Git)
├── report/                  # báo cáo LaTeX (tệp `Báo cáo đồ án GK - Lập trình Python.tex` và thư mục chapters/)
├── tests/                   # kiểm thử tự động bằng pytest
├── pipeline.py              # kiểm tra, làm sạch, phân tích, vẽ 5 biểu đồ
├── train_model.py           # huấn luyện và đánh giá mô hình hồi quy
├── melbourne_data_analysis_pipeline.ipynb  # notebook triển khai theo từng bước
├── requirements.txt         # thư viện cần cài (khoảng phiên bản)
├── requirements-lock.txt    # phiên bản chính xác đã dùng để chạy lại
├── pytest.ini
└── README.md
```

## 3. Cài đặt môi trường

Tạo môi trường ảo (tùy chọn nhưng nên dùng):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Số đo của mô hình có thể lệch rất nhẹ giữa các phiên bản thư viện (xem mục 8). Muốn cài đúng phiên bản đã dùng khi chạy lại đồ án, thay `requirements.txt` bằng `requirements-lock.txt` (đã kiểm tra với Python 3.13).

## 4. Chạy pipeline Python

Từ thư mục dự án:

```powershell
python pipeline.py
```

Kết quả được lưu trong `outputs/`: dữ liệu sau làm sạch (`melb_data_clean.csv`, 13.580 dòng × 27 cột; thiếu thêm 26 ô `Landsize`, 8 ô `BuildingArea` và 34 ô `Bathroom` so với trước khi loại ngoại lai), báo cáo chất lượng, bảng phân tích (loại nhà, vùng, số phòng, suburb, tương quan, tháng bán), 5 nhận xét tự động (`findings.txt`), 5 biểu đồ phân tích (`01`–`05`), 2 biểu đồ chất lượng dữ liệu (`08`, `09`) và `report_metrics.csv` (các số bổ sung của báo cáo, kèm cách tính).

## 5. Chạy kiểm thử

```powershell
python -m pytest
```

Bộ kiểm thử trong `tests/` chạy trong vài giây và không huấn luyện lại mô hình. Nó kiểm tra bước làm sạch (số dòng, các quy tắc ngoại lai, cờ `*_was_missing`, cột ngày), tỷ lệ chia tập 64% / 16% / 20%, các hàm đánh giá và kiểm tra quá khớp, cùng tính nhất quán giữa các tệp kết quả trong `outputs/`. Chạy lệnh này trước khi đưa thay đổi lên Git.

## 6. Chạy notebook

Mở `melbourne_data_analysis_pipeline.ipynb` trong VS Code/Jupyter và chạy lần lượt các cell từ trên xuống. Notebook sử dụng đường dẫn `data/melb_data.csv` tương đối với thư mục dự án.

## 7. Quy tắc xử lý dữ liệu

- Dữ liệu gốc không bị ghi đè.
- Dòng trùng hoàn toàn và dòng không có giá hợp lệ bị loại; với tệp này cả hai đều bằng 0 nên không dòng nào bị xóa (vẫn 13.580 dòng).
- `Date` chuyển sang kiểu datetime với `dayfirst=True` theo định dạng ngày/tháng/năm của bộ dữ liệu.
- Giá bán không dương, số phòng không dương, khoảng cách/diện tích đất âm và diện tích xây dựng không dương được đánh dấu thiếu; `YearBuilt < 1700` được xem là không hợp lý và chuyển thành thiếu. Trên tệp này có 17 dòng `BuildingArea <= 0` và 1 dòng `YearBuilt < 1700` bị ảnh hưởng.
- Ngoại lai về diện tích và phòng tắm được chuyển thành thiếu (không xóa dòng): `Landsize > 10.000` m² (26 dòng), `BuildingArea > 1.000` m² (8 dòng) và `Bathroom == 0` (34 dòng). Ngưỡng là lựa chọn của dự án (hằng số `MAX_LANDSIZE`, `MAX_BUILDING_AREA` trong `pipeline.py`). `Landsize = 0` được giữ vì phần lớn là căn hộ.
- `CouncilArea` thiếu (1.369 dòng) được điền `Unknown`; thêm `SaleYear` và `SaleMonth` từ `Date`.
- `BuildingArea` và `YearBuilt` có tỷ lệ thiếu lớn; mặc định không nội suy tuỳ tiện trong pipeline EDA. Các cột này vẫn còn NaN để tránh tạo ra số liệu giả cho thống kê mô tả.
- Thêm cờ `*_was_missing` cho một số cột quan trọng trước bước sửa giá trị không hợp lý. Giá trị bị chuyển thành thiếu do kiểm tra tính hợp lệ không nhất thiết là giá trị thiếu ban đầu; nếu dùng cho mô hình ML, cần tách riêng hai loại cờ.
- Giá cao không bị loại: 612 dòng vượt ngưỡng IQR (2.350.000 AUD) đều là giao dịch có thể có thật; mô hình dùng `log1p(Price)` để giảm ảnh hưởng của đuôi phải.
- Hệ quả của việc loại ngoại lai diện tích: Pearson của `BuildingArea` với giá tăng từ 0,09 lên 0,56 và của `Landsize` từ 0,04 lên 0,13 (xem báo cáo, Phép 6). Sai số mô hình gần như không đổi (MAE test của Random Forest là 155.236 AUD khi chưa loại ngoại lai và 156.203 AUD khi đã loại, theo `outputs/doi_chung/`).

## 8. Phạm vi phân tích

- Kiểm tra schema, kiểu dữ liệu, số lượng thiếu, dòng trùng và giá trị không hợp lý.
- Thống kê mô tả giá bán và các thuộc tính số.
- So sánh giá theo loại bất động sản, vùng, số phòng, suburb và tháng bán.
- Phân tích tương quan giá với khoảng cách, số phòng và một số thuộc tính số.
- 5 biểu đồ: histogram, boxplot, bar chart, scatter plot và correlation heatmap.
- Ghi 5 nhận xét dựa trên kết quả tính toán thực tế vào `outputs/findings.txt`.

Lưu ý: đây là bộ dữ liệu snapshot giai đoạn 2016–2017 và thuộc thị trường Melbourne, Australia; không đại diện cho giá nhà hiện tại hoặc thị trường Việt Nam.

## 9. Xây dựng mô hình hồi quy

Sau khi chạy `pipeline.py` để tạo `outputs/melb_data_clean.csv`, chạy:

```powershell
python -m pip install -r requirements.txt
python train_model.py
```

`train_model.py` thực hiện các bước:

1. Chọn `Price` làm biến mục tiêu; dùng các thuộc tính nhà ở và vị trí làm biến đầu vào.
2. Chia ngẫu nhiên với `random_state=42`: tách 20% làm tập test, rồi tách 20% phần còn lại làm tập validation (train 8.691, validation 2.173, test 2.716 dòng).
3. So sánh baseline dự đoán giá trung vị, Ridge Regression và Random Forest.
4. Đặt bước điền giá trị thiếu và One-Hot Encoding bên trong `Pipeline` để chỉ học quy tắc tiền xử lý từ tập train.
5. Huấn luyện trên `log1p(Price)` vì giá có phân phối lệch phải; dự đoán được chuyển ngược về AUD trước khi tính metric.
6. So sánh bằng MAE, RMSE và R²; chọn mô hình có MAE thấp nhất trên tập validation, huấn luyện lại mô hình đã chọn trên train + validation rồi chấm trên tập test một lần.
7. Kiểm tra quá khớp và dưới khớp: đo sai số trên tập train rồi đặt cạnh sai số trên tập chưa dùng để học, in thành bảng trên màn hình và ghi vào `outputs/model_fit_check.csv`.

**Đọc bảng kiểm tra quá khớp.** Cột `MAE_eval_chia_MAE_train` gần 1 mà sai số vẫn cao cho thấy mô hình quá đơn giản (dưới khớp); cột này lớn cho thấy mô hình khớp tập train tốt hơn dữ liệu mới (có khoảng cách quá khớp). Kết quả hiện tại:

| Mô hình | MAE train | MAE validation hoặc test | Tỷ lệ | Nhận định |
|---|---|---|---|---|
| Ridge | 228.926 | 231.137 (validation) | 1,01 | Dưới khớp |
| Random Forest | 75.477 | 154.865 (validation) | 2,05 | Còn khoảng cách giữa train và validation |
| Random Forest huấn luyện lại trên train + validation | 73.066 | 156.203 (test) | 2,14 | Vẫn có MAE thấp nhất trên validation |

Các siêu tham số đang đặt cố định và chưa kiểm định chéo, nên khoảng cách của Random Forest có thể còn thu hẹp được khi tinh chỉnh.

Kết quả được ghi vào:

- `models/best_pipeline.joblib`: mô hình tốt nhất đã fit trong môi trường Python hiện tại.
- `models/model_metadata.json`: tên mô hình, các thuộc tính và cấu hình đánh giá.
- `outputs/model_validation_metrics.csv`: so sánh mô hình trên tập validation.
- `outputs/model_metrics.csv`: kết quả trên tập test của mô hình đã chọn và baseline.
- `outputs/test_predictions.csv`: giá thực tế, giá dự đoán và residual của tập test.
- `outputs/06_actual_vs_predicted.png`, `outputs/07_residual_distribution.png`: biểu đồ chẩn đoán.
- `outputs/model_fit_check.csv`: sai số trên tập train đặt cạnh sai số trên validation hoặc test, dùng để kiểm tra quá khớp và dưới khớp.
- `outputs/model_error_metrics.csv`: MAPE, sai số theo phân khúc, vùng, loại nhà và kích thước các tập (các số bổ sung của báo cáo, kèm cách tính).

**Lưu ý đánh giá:** bộ dữ liệu là snapshot lịch sử Melbourne, không phải dữ liệu giá hiện tại. Chia ngẫu nhiên phù hợp để thiết lập baseline cho bài tập, nhưng sai số có thể lạc quan hơn so với dự đoán giai đoạn sau; nếu mục tiêu là dự đoán cho giai đoạn tương lai, nên dùng chia tập theo thời gian. Mô hình hiện dùng cả `Method`, `Postcode` và `Bedroom2`; nếu định giá trước khi bán, nên thử loại `Method`. Số đo có thể lệch nhẹ giữa các phiên bản thư viện: Random Forest không đổi, còn MAE validation của Ridge là 231.160 trong `outputs/model_validation_metrics.csv` và 231.137 trong `outputs/model_fit_check.csv` (lệch khoảng 0,01%) vì hai tệp được tạo ở hai môi trường. Báo cáo dùng số trong `outputs/`; muốn tái lập đúng một bộ số, cài theo `requirements-lock.txt` và chạy lại cả `pipeline.py` lẫn `train_model.py`. Hãy chạy `train_model.py` trong đúng môi trường của ứng dụng để tự sinh file `.joblib`; không nên chuyển file mô hình đã lưu giữa các phiên bản thư viện khác nhau mà chưa kiểm tra tương thích.

## 10. Báo cáo LaTeX

Thư mục `report/` chỉ chứa tài liệu LaTeX; hình và số liệu của báo cáo đều lấy từ `outputs/` (hình qua `\graphicspath{{../outputs/}}`). Chạy `pipeline.py` và `train_model.py` trước để có `outputs/` đầy đủ, rồi biên dịch bằng XeLaTeX:

```powershell
cd report
xelatex "Báo cáo đồ án GK - Lập trình Python.tex"
xelatex "Báo cáo đồ án GK - Lập trình Python.tex"
```

Mỗi bảng, hình, số trong báo cáo có dòng "Nguồn" trỏ tới một tệp trong `outputs/` hoặc một mã trong `outputs/report_metrics.csv` (do `pipeline.py` ghi) và `outputs/model_error_metrics.csv` (do `train_model.py` ghi). `outputs/doi_chung/` là kết quả mô hình trước khi loại ngoại lai. Bảng đối chiếu ở Phụ lục B của báo cáo.
