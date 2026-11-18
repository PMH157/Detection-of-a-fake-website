# Hướng dẫn sử dụng trang web kiểm thử

Trang web dùng mô hình **Random Forest đã chọn trong đồ án** để đoán một website là **giả mạo** hay **hợp lệ**.
Đây là phần kiểm thử: không thay đổi thí nghiệm, kết quả hay quyết định cuối của đồ án.

---

## 1. Chuẩn bị

- **Python 3.10 trở lên** (đã chạy thử trên Python 3.13).
- Thư mục `web` phải nằm **trong thư mục gốc của đồ án**, cùng cấp với `src/`, `results/`, `data/`:

```
Detection-of-a-fake-website/
├─ data/          # phishing.csv (tự tải nếu chưa có)
├─ results/       # decision.json: siêu tham số đã chọn
├─ src/           # code của đồ án (trang web dùng lại hàm đọc dữ liệu)
├─ requirements.txt
└─ web/           # thư mục này
```

- Cần **Internet** khi: chạy lần đầu mà chưa có `data/phishing.csv`, hoặc dùng tab **Kiểm tra bằng URL**.

---

## 2. Cài đặt bằng môi trường ảo (chỉ làm lần đầu)

Mở terminal tại **thư mục gốc của đồ án**.

### Windows (PowerShell)

```bash
python -m venv .venv
```
```bash
.venv\Scripts\Activate.ps1
```
```bash
pip install -r requirements.txt -r web/requirements.txt
```

Nếu PowerShell báo *"running scripts is disabled"*, chạy lệnh sau một lần rồi kích hoạt lại:

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Dùng CMD thì kích hoạt bằng `.venv\Scripts\activate.bat`.

### macOS / Linux

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt -r web/requirements.txt
```

---

## 3. Chạy trang web

Kích hoạt môi trường ảo (bước 2), rồi chạy tại thư mục gốc của đồ án:

```bash
streamlit run web/app.py
```

- Trình duyệt tự mở `http://localhost:8501`; nếu không, tự dán địa chỉ này vào trình duyệt.
- Lần đầu mất khoảng 10–20 giây để huấn luyện mô hình; các lần thao tác sau nhanh hơn vì mô hình được lưu trong bộ nhớ.
- Tắt trang: nhấn **Ctrl + C** trong terminal. Thoát môi trường ảo: gõ `deactivate`.
- Môi trường ảo **không mất** khi thoát hay tắt máy. Lần sau chỉ cần kích hoạt lại và chạy lệnh `streamlit run web/app.py`.

---

## 4. Các chức năng

### 🌐 Tab "Kiểm tra bằng URL"

1. Dán đường link vào ô (có thể bỏ `http://`, ví dụ `vnexpress.net`).
2. Nhấn **Kiểm tra**.
3. Trang hiện:
   - **Kết luận**: Website HỢP LỆ (xanh) hoặc Website GIẢ MẠO (đỏ), ngưỡng quyết định 50%.
   - **Xác suất giả mạo** trên đồng hồ và thanh mức độ rủi ro (An toàn → Nghi ngờ → Giả mạo).
   - **Số đặc trưng đã dùng** và **F1 trên tập test** của mô hình tương ứng.
   - **Bảng đặc trưng**: giá trị của từng đặc trưng và căn cứ (ví dụ "62 ký tự", "40% của 120 liên kết trỏ ra ngoài").

Trang tự đo **18 đặc trưng** từ các nguồn sau:

| Nguồn | Đặc trưng |
|---|---|
| Chuỗi URL | dùng địa chỉ IP, độ dài URL, ký tự `@`, dấu gạch ngang trong tên miền, số tên miền con, cổng |
| Mã HTML của trang | tài nguyên từ tên miền khác, liên kết `<a>` trỏ ra ngoài, liên kết trong thẻ meta/script/link, nơi gửi form, form gửi về email, đổi thanh trạng thái khi rê chuột, chặn chuột phải, pop-up nhập liệu, iframe ẩn |
| Kết nối mạng | chứng chỉ SSL (https hợp lệ không), bản ghi DNS, tuổi tên miền (tra qua rdap.org) |

**Vì sao không dùng đủ 30 đặc trưng?**

- 5 đặc trưng cần dịch vụ bên ngoài đã ngừng hoặc phải trả phí: lưu lượng truy cập, PageRank, Google index,
  số liên kết trỏ tới trang, báo cáo thống kê.
- 7 đặc trưng có **chiều mã hóa trong dữ liệu ngược với quy tắc gốc** (dịch vụ rút gọn link, chuyển hướng bằng `//`,
  thời hạn đăng ký tên miền, favicon, chữ "https" trong tên miền, URL bất thường, số lần chuyển hướng).
  Tự đo theo quy tắc rồi đưa vào sẽ làm mô hình hiểu ngược.
- Đặc trưng nào không đo được (trang chặn truy cập, không tra được tuổi tên miền, trang không có liên kết…) cũng bị bỏ qua.

Trang **không điền giá trị giả** cho đặc trưng thiếu. Thay vào đó, nó **huấn luyện lại mô hình với đúng siêu tham số đã chọn**
trên các đặc trưng đo được, rồi báo F1 test của mô hình đó (18 đặc trưng: F1 ≈ 0.933; đủ 30 đặc trưng: 0.9444).

### 🔍 Tab "Kiểm tra mẫu trong tập test"

- Chọn một mẫu trong 1.170 mẫu test bằng số thứ tự, hoặc nhấn **🎲 Mẫu ngẫu nhiên**.
- Thẻ kết quả so sánh dự đoán với **nhãn thật** (Dự đoán đúng / Dự đoán sai).
- Có thể sửa từng ô trong **30 đặc trưng** để thử "nếu… thì…". Khi đã sửa, trang không còn nhãn thật để so sánh.
- **✅ Toàn bộ hợp lệ**: đặt cả 30 đặc trưng về giá trị hợp lệ (1).

Quy ước giá trị của bộ dữ liệu: **1 = hợp lệ, 0 = nghi ngờ, -1 = dấu hiệu giả mạo**.

### 📊 Tab "Hiệu năng mô hình"

- F1, Precision, Recall trên tập test (0.9444 / 0.9622 / 0.9272, khớp với báo cáo).
- Ma trận nhầm lẫn.
- Bảng siêu tham số đã chọn (300 cây, độ sâu 14, max_features 0.05, …).
- 10 đặc trưng quan trọng nhất của mô hình.

---

## 5. Cấu trúc thư mục `web`

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit: 3 tab, huấn luyện và dự đoán |
| `url_features.py` | Trích xuất đặc trưng từ một URL thật |
| `test_url_features.py` | Kiểm tra quy tắc trích xuất (không cần mạng) |
| `static/style.css` | CSS giao diện sáng: nền pastel, tab dạng viên, thẻ trắng bo góc |
| `static/hero.html` | Phần tiêu đề (HTML/CSS/JS): số liệu đếm tăng, bong bóng pastel trên canvas |
| `static/result.html` | Thẻ kết quả (HTML/CSS/JS): đồng hồ xác suất, thanh mức độ rủi ro |
| `.streamlit/config.toml` | Luôn dùng giao diện sáng, kể cả khi máy bật chế độ tối |
| `requirements.txt` | Thư viện riêng cho trang web (Streamlit) |
| `README.md` | Giới thiệu ngắn |
| `HUONG_DAN.md` | File hướng dẫn này |

Chạy kiểm tra (tại thư mục gốc của đồ án, đã kích hoạt môi trường ảo):

```bash
pytest web
```

---

## 6. Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `streamlit: command not found` / không nhận lệnh | Chưa kích hoạt môi trường ảo hoặc chưa cài thư viện → làm lại bước 2 |
| `ModuleNotFoundError: experiment` | Thư mục `web` không nằm trong thư mục gốc của đồ án → giải nén lại đúng chỗ |
| Báo lỗi tải dữ liệu lần đầu | Kiểm tra Internet; hoặc chép sẵn `phishing.csv` vào `data/` |
| "Không tải được trang …" ở tab URL | Trang chặn truy cập tự động, cần đăng nhập hoặc đã ngừng hoạt động; trang vẫn dự đoán bằng các đặc trưng còn lại |
| `Port 8501 is already in use` | Đang có một trang khác chạy → tắt nó, hoặc chạy `streamlit run web/app.py --server.port 8502` |
| Lỗi đường dẫn sau khi di chuyển/đổi tên thư mục đồ án | Xóa thư mục `.venv` rồi làm lại bước 2 |

---

## 7. Lưu ý

- Kết quả **chỉ để tham khảo**. Bộ dữ liệu UCI được thu thập khoảng 2012–2015 với các quy tắc đặc trưng đơn giản,
  nên độ chính xác trên website hiện nay có thể thấp hơn số đo trên tập test.
- Khi thử link lừa đảo thật (ví dụ lấy từ PhishTank, OpenPhish): trang chỉ tải mã HTML, **không chạy JavaScript**,
  nhưng **đừng mở các link đó bằng trình duyệt**.
- Không nhập link chứa thông tin cá nhân (mã đăng nhập, token…), vì trang sẽ gửi yêu cầu tới địa chỉ đó.
