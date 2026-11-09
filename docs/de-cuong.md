# Đề cương & đăng ký đề — Detection of a Fake Website

| Mục | Nội dung |
|---|---|
| Tên đề tài | Tối ưu siêu tham số Random Forest có ràng buộc kích thước mô hình cho bài toán phát hiện website giả mạo (phishing) |
| Bối cảnh / vấn đề | Bộ lọc phishing chạy phía client (extension trình duyệt, thiết bị biên) cần vừa **bắt đúng nhiều site giả** vừa **nhỏ, nhanh**. RF mặc định của scikit-learn cho chất lượng tốt nhưng các cây mọc tới hết độ sâu → mô hình nặng. Câu hỏi: có cấu hình RF nào giữ (hoặc tăng) F1 mà nằm trong ngân sách dung lượng không, và thuật toán search nào tìm ra nó hiệu quả nhất với cùng số lần đánh giá? |
| Dataset / nguồn | UCI Phishing Websites (id 327), benchmark công bố, CC BY 4.0, public được trên GitHub. Xem `data/README.md`. |
| Decision variables x | `n_estimators` ∈ [10, 300] (int, log), `max_depth` ∈ [2, 30] (int), `min_samples_leaf` ∈ [1, 20] (int), `max_features` ∈ [0.05, 1.0] (tỉ lệ đặc trưng) |
| Objective f(x) | max F1 trung bình 3-fold CV trên tập train (lớp dương = phishing) |
| Constraints / trade-off | `size_MB(x) ≤ S_max = 1.0 MB` (dung lượng pickle của mô hình), xử lý bằng penalty: `J(x) = F1_cv(x) − λ·max(0, size(x) − S_max)/S_max`, λ = 1. Trade-off F1 ↔ size được thể hiện bằng Pareto front. |
| Baseline | (1) RF mặc định của scikit-learn (không search); (2) Random Search cùng budget |
| Phương pháp tối ưu chính | TPE — Bayesian optimization (Optuna `TPESampler`) |
| Metric | Search: J(x), F1_cv. Báo cáo cuối: F1/accuracy/recall trên test, size MB, latency ms/1000 mẫu, thời gian search, tỉ lệ khả thi |
| Budget / điều kiện dừng | 40 lần đánh giá J(x) / mỗi lần chạy (mỗi lần = 3 lần fit RF); 10 seed cho mỗi phương pháp |
| Sản phẩm tuần 5 | Repo public + README + `results/*.csv` + 3 hình + báo cáo 8–12 trang + slide 5–8 trang + tag v1.0 |

## Tối ưu nằm ở đâu? (trả lời câu hỏi "tối ưu chỗ nào")

Huấn luyện RF tự nó đã là một bài toán tối ưu (mỗi cây chọn ngưỡng chia tốt nhất theo Gini) — nhưng đó **không** phải phần đóng góp của đồ án. Phần tối ưu của đồ án là **tầng ngoài**:

```
x* = argmax_{x ∈ Ω}  J(x) = F1_cv(x) − λ · max(0, size(x) − S_max) / S_max
```

- J(x) là **hàm hộp đen**: không có đạo hàm, mỗi lần tính tốn 3 lần fit RF, có nhiễu (do bootstrap ngẫu nhiên) → không dùng được gradient descent → dùng **search có budget** (Random, TPE).
- Không gian Ω trộn biến nguyên và liên tục.
- Ràng buộc size **active**: RF mặc định vi phạm ngân sách (xem kết quả), nên bài toán không tầm thường — "cây càng nhiều càng sâu càng tốt" không còn đúng.

## Kế hoạch 5 tuần

| Tuần | Việc | Checkpoint commit |
|---|---|---|
| 1 | Chọn dataset, viết formulation (file này), dựng repo | `docs: formulation` |
| 2 | Loader + dedup + split; baseline RF mặc định | `feat: data + baseline` |
| 3 | Random Search + TPE với objective có penalty; logging trial | `feat: search + logging` |
| 4 | Chạy 10 seed, sinh bảng + hình, sensitivity (đổi S_max) | `results: main experiment` |
| 5 | Báo cáo, slide, README Key Results, tag `v1.0` | `docs: report`, tag `v1.0` |
