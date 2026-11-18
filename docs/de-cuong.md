# Đề cương & đăng ký đề — Detection of a Fake Website

| Mục | Nội dung |
|---|---|
| Tên đề tài | Tối ưu siêu tham số Random Forest theo từng siêu tham số (coordinate search) cho bài toán phát hiện website giả mạo (phishing) |
| Bối cảnh / vấn đề | Random Forest phát hiện phishing tốt trên đặc trưng URL/tên miền, nhưng chất lượng phụ thuộc nhiều siêu tham số đang để mặc định. Câu hỏi: thay lần lượt từng siêu tham số thì F1 tăng bao nhiêu, mỗi siêu tham số đóng góp gì, và cái giá về dung lượng / thời gian dự đoán là bao nhiêu? |
| Dataset / nguồn | UCI Phishing Websites (id 327), benchmark công bố, CC BY 4.0, public được trên GitHub. Xem `data/README.md`. |
| Decision variables x | 10 siêu tham số RF trên lưới giá trị: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` (chính); `criterion`, `bootstrap`, `min_samples_split`, `max_leaf_nodes`, `class_weight`, `ccp_alpha` (phụ) |
| Objective f(x) | max F1 trung bình 3-fold CV trên tập train (lớp dương = phishing) |
| Constraints / trade-off | x thuộc lưới Ω; ngân sách đánh giá hữu hạn (40 lần cho 4 siêu tham số chính, +24 lần cho 6 siêu tham số phụ). Không giới hạn dung lượng (chấp nhận), nhưng dung lượng và latency được đo, báo cáo như trade-off. |
| Baseline | (1) RF mặc định của scikit-learn; (2) Random Search; (3) TPE — cùng 4 siêu tham số chính, cùng 40 lần đánh giá |
| Phương pháp tối ưu chính | Coordinate search (one-factor-at-a-time): thay lần lượt từng siêu tham số, mỗi tham số một lần |
| Metric | Search: F1_cv. Báo cáo cuối: F1/precision/recall trên test, dung lượng MB, latency ms/1000 mẫu, thời gian search; Wilcoxon theo cặp seed |
| Budget / điều kiện dừng | Mỗi bước thử hết lưới của một siêu tham số rồi dừng; 1 lượt qua 10 siêu tham số (65 lần đánh giá kể cả điểm xuất phát; mỗi lần = 4 lần fit RF); 10 seed |
| Sản phẩm tuần 5 | Repo public + README + `results/*` + hình + báo cáo 8–12 trang + slide 5–8 trang + tag v1.0 |

## Tối ưu nằm ở đâu? (trả lời câu hỏi "tối ưu chỗ nào")

Huấn luyện RF tự nó đã là một bài toán tối ưu (mỗi cây chọn ngưỡng chia tốt nhất theo Gini) — nhưng đó **không** phải phần đóng góp của đồ án. Phần tối ưu của đồ án là **tầng ngoài**:

```
x* = argmax_{x ∈ Ω}  F1_cv(x)
```

- F1_cv(x) là **hàm hộp đen**: không có đạo hàm, mỗi lần tính tốn 4 lần fit RF, có nhiễu (do bootstrap ngẫu nhiên) → không dùng được gradient descent → dùng **tìm kiếm có ngân sách**.
- Ω trộn biến nguyên, liên tục và phân loại (criterion, bootstrap, class_weight).
- Coordinate search chia bài toán 10 chiều thành 10 bài toán 1 chiều giải tuần tự — dễ giải thích từng siêu tham số đóng góp gì, nhưng phụ thuộc thứ tự và bỏ qua tương tác.

## Kế hoạch 5 tuần

| Tuần | Việc | Checkpoint commit |
|---|---|---|
| 1 | Chọn dataset, viết formulation (file này), dựng repo | `docs: formulation` |
| 2 | Loader + dedup + split; baseline RF mặc định | `feat: data + baseline` |
| 3 | Random Search + TPE; coordinate search V0–V4; logging trial | `feat: search + logging` |
| 4 | Coordinate search V5–V10, 10 seed, bảng + hình, kiểm tra overfitting | `results: main experiment` |
| 5 | Báo cáo, slide, README Key Results, tag `v1.0` | `docs: report`, tag `v1.0` |
