# Lưu trữ: phiên bản cũ có ngân sách dung lượng 1 MB

Đây là kết quả của phiên bản trước, khi bài toán có ràng buộc `size ≤ 1 MB` (λ = 1).
Quyết định cuối của đồ án đã đổi sang tối ưu từng siêu tham số một **không giới hạn dung lượng**
(xem README chính); thư mục này chỉ giữ lại để tham khảo, không dùng trong báo cáo mới.

- `results/`, `figures/`: Default / Random / TPE với S_max = 1 MB, độ nhạy S_max, ablation, độ bền, scalability, kiểm tra overfitting.
- `ofat/`, `ofat_extra/`: tối ưu từng siêu tham số một **có** ngân sách 1 MB (bản thử nghiệm).
- `try_config/`: thử cấu hình thủ công 50 cây / sâu 15 so với cấu hình TPE cũ.
- `src/extra_experiments.py`: script của các thí nghiệm mở rộng cũ (để chạy lại phải chép về `src/` và đọc kết quả từ thư mục này).
- Chạy lại thí nghiệm chính cũ: `python src/experiment.py --lam 1 --out results_trial/budget_1mb/results`.
