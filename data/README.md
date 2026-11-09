# Dữ liệu

**Phishing Websites** — UCI Machine Learning Repository, dataset id 327.
Mohammad, R., Thabtah, F. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI. https://doi.org/10.24432/C51W2X — giấy phép CC BY 4.0.

- File dùng: `Training Dataset.arff` trong zip chính thức (11 055 dòng, 30 đặc trưng, nhãn `Result`).
- Đặc trưng: 30 thuộc tính đã mã hóa sẵn về {-1, 0, 1} (độ dài URL, có IP trong URL, ký tự `@`, tuổi tên miền, SSL, số link ngoài, iframe, PageRank...). Không cần truy cập website thật.
- Nhãn: `Result = -1` → phishing (mã hóa thành **1**, lớp dương), `Result = 1` → hợp lệ (**0**).

## Tải về

Không cần tải tay: `src/experiment.py` tự tải zip từ UCI ở lần chạy đầu và lưu `data/phishing.csv` (file này bị `.gitignore`).

## Tiền xử lý

1. Bỏ các dòng trùng lặp hoàn toàn: **11 055 → 5 849 dòng**. Gần một nửa dataset là bản sao; nếu giữ lại, cùng một mẫu sẽ nằm ở cả train và test → điểm số bị thổi phồng (data leakage).
2. Chia train/test 80/20 phân tầng, `random_state=42`, dùng chung cho mọi phương pháp. Test set chỉ dùng **một lần** sau khi search đã chọn xong cấu hình.
3. Không cần chuẩn hóa (Random Forest không nhạy với scale; đặc trưng đã rời rạc).
