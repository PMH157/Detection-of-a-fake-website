# Trang web kiểm thử (Streamlit + HTML/CSS/JS)

Phần kiểm thử, không ảnh hưởng tới thí nghiệm và quyết định cuối. Trang huấn luyện lại đúng mô hình đã chọn
(`results/decision.json`) trên tập train cố định rồi dự đoán các mẫu trong tập test.

```bash
pip install -r requirements.txt -r web/requirements.txt
streamlit run web/app.py
```

- `app.py`: giao diện Streamlit, 3 tab: kiểm tra bằng URL, kiểm tra mẫu trong tập test (sửa được 30 đặc trưng), hiệu năng.
- `url_features.py`: trích xuất 18 đặc trưng từ URL thật (chuỗi URL, HTML của trang, chứng chỉ SSL, DNS, tuổi tên miền qua RDAP).
  Đặc trưng không đo được bị loại và mô hình được huấn luyện lại (cùng siêu tham số đã chọn) trên các đặc trưng còn lại.
- `test_url_features.py`: kiểm tra quy tắc trích xuất, chạy `pytest web` (không cần mạng).
- `static/style.css`: CSS giao diện sáng (nền pastel, tab dạng viên, thẻ trắng bo góc, hiệu ứng khi rê chuột).
- `static/hero.html`: phần tiêu đề HTML/CSS/JS (chữ gõ dần, số liệu đếm tăng, bong bóng pastel trôi trên canvas).
- `static/result.html`: thẻ kết quả HTML/CSS/JS (đồng hồ xác suất, thanh mức độ rủi ro có hiệu ứng).
- `.streamlit/config.toml`: luôn dùng giao diện sáng, kể cả khi máy đang bật chế độ tối.

Chế độ URL cần Internet (tải trang, tra DNS, tra RDAP tại rdap.org). Kết quả chỉ để tham khảo: bộ dữ liệu UCI
thu thập khoảng 2012–2015, và 12/30 đặc trưng không đo được hoặc có chiều mã hóa ngược với quy tắc gốc.
