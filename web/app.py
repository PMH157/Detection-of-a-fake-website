"""Trang web kiểm thử: mô hình Random Forest đã chọn (V10) dự đoán website giả mạo.

Chạy: streamlit run web/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment import SPLIT_SEED, TEST_SIZE, load_data  # noqa: E402
from step_summary import to_kwargs  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from url_features import USED, extract  # noqa: E402

STATIC = Path(__file__).parent / "static"

# 1 = hợp lệ, 0 = nghi ngờ, -1 = dấu hiệu giả mạo (quy ước của bộ dữ liệu UCI)
LABELS = {
    "having_IP_Address": "Dùng địa chỉ IP thay tên miền", "URL_Length": "Độ dài URL",
    "Shortining_Service": "Dùng dịch vụ rút gọn link", "having_At_Symbol": "URL có ký tự @",
    "double_slash_redirecting": "Chuyển hướng bằng //", "Prefix_Suffix": "Tên miền có dấu gạch ngang",
    "having_Sub_Domain": "Số tên miền con", "SSLfinal_State": "Chứng chỉ SSL",
    "Domain_registeration_length": "Thời hạn đăng ký tên miền", "Favicon": "Favicon từ tên miền khác",
    "port": "Cổng không chuẩn", "HTTPS_token": "Chữ 'https' trong tên miền",
    "Request_URL": "Tài nguyên tải từ tên miền khác", "URL_of_Anchor": "Liên kết <a> trỏ ra ngoài",
    "Links_in_tags": "Liên kết trong thẻ meta/script", "SFH": "Nơi gửi form (SFH)",
    "Submitting_to_email": "Form gửi về email", "Abnormal_URL": "URL bất thường so với WHOIS",
    "Redirect": "Số lần chuyển hướng", "on_mouseover": "Đổi thanh trạng thái khi rê chuột",
    "RightClick": "Chặn chuột phải", "popUpWidnow": "Cửa sổ pop-up", "Iframe": "Dùng iframe ẩn",
    "age_of_domain": "Tuổi tên miền", "DNSRecord": "Bản ghi DNS", "web_traffic": "Lưu lượng truy cập",
    "Page_Rank": "PageRank", "Google_Index": "Được Google lập chỉ mục",
    "Links_pointing_to_page": "Số liên kết trỏ tới trang", "Statistical_report": "Có trong báo cáo thống kê",
}
VALUE_TEXT = {1: "1 · hợp lệ", 0: "0 · nghi ngờ", -1: "-1 · giả mạo"}


@st.cache_resource(show_spinner="Đang tải dữ liệu…")
def data():
    X, y, _ = load_data()
    cols = pd.read_csv(ROOT / "data" / "phishing.csv", nrows=0).columns[:-1].tolist()
    Xtr, Xte, ytr, yte = train_test_split(pd.DataFrame(X, columns=cols), y, test_size=TEST_SIZE, stratify=y,
                                          random_state=SPLIT_SEED)
    return cols, Xtr, Xte, ytr, yte, json.loads((ROOT / "results" / "decision.json").read_text())


@st.cache_resource(show_spinner="Đang huấn luyện mô hình với siêu tham số đã chọn…")
def model_for(features):
    """Siêu tham số đã chọn (decision.json), huấn luyện trên tập train với các đặc trưng cho trước."""
    _, Xtr, Xte, ytr, yte, dec = data()
    f = list(features)
    m = RandomForestClassifier(**to_kwargs(dec["params"]), n_jobs=-1, random_state=dec["seed"]).fit(Xtr[f].to_numpy(), ytr)
    p = m.predict(Xte[f].to_numpy())
    return m, {"f1": f1_score(yte, p), "precision": precision_score(yte, p), "recall": recall_score(yte, p)}


def card(prob, truth=None):
    """Kết quả dự đoán: HTML + CSS + JS (đồng hồ đo có hiệu ứng)."""
    html = (STATIC / "result.html").read_text(encoding="utf8")
    data = {"prob": float(prob), "truth": None if truth is None else int(truth)}
    components.html(html.replace("__DATA__", json.dumps(data)), height=206)


st.set_page_config(page_title="Phát hiện website giả mạo", page_icon="🛡️", layout="wide")
st.markdown(f"<style>{(STATIC / 'style.css').read_text(encoding='utf8')}</style>", unsafe_allow_html=True)
if not (ROOT / "data" / "phishing.csv").exists():
    st.info("Lần chạy đầu sẽ tải bộ dữ liệu UCI Phishing Websites (cần Internet).")
cols, Xtr, Xte, _, yte, dec = data()
model, main_metrics = model_for(tuple(cols))
hero = (STATIC / "hero.html").read_text(encoding="utf8")
components.html(hero.replace("__F1__", f"{main_metrics['f1']:.4f}").replace("__N__", str(len(Xtr) + len(Xte))), height=192)

tab_url, tab_test, tab_perf = st.tabs(["🌐 Kiểm tra bằng URL", "🔍 Kiểm tra mẫu trong tập test", "📊 Hiệu năng mô hình"])

with tab_url:
    with st.form("url_form"):
        url = st.text_input("Dán URL của trang web cần kiểm tra", placeholder="https://example.com/login")
        go = st.form_submit_button("Kiểm tra", type="primary")
    if go and url.strip():
        with st.spinner("Đang tải trang, tra DNS và thông tin tên miền…"):
            try:
                st.session_state.url_res = extract(url)
            except ValueError as e:
                st.session_state.pop("url_res", None)
                st.error(str(e))
    if "url_res" in st.session_state:
        f, why, info = st.session_state.url_res
        used = tuple(c for c in USED if f[c] is not None)
        m_url, met = model_for(used)
        prob = m_url.predict_proba(pd.DataFrame([f])[list(used)].to_numpy())[0, 1]
        st.markdown(f"### Kết quả cho `{info['url']}`")
        card(prob)
        st.caption(f"Mô hình dùng {len(used)}/30 đặc trưng đo được, cùng siêu tham số đã chọn · "
                   f"F1 trên tập test của mô hình này: {met['f1']:.4f} (mô hình đủ 30 đặc trưng: 0.9444)")
        if "fetch_error" in info:
            st.warning(f"Không tải được trang ({info['fetch_error']}), nên các đặc trưng lấy từ HTML và SSL bị bỏ qua.")
        st.dataframe(pd.DataFrame([{"Đặc trưng": LABELS[c],
                                    "Giá trị": "không đo được" if f[c] is None else VALUE_TEXT[f[c]],
                                    "Căn cứ": why.get(c, "")} for c in USED]),
                     hide_index=True, use_container_width=True)
        with st.expander("12 đặc trưng không dùng ở chế độ URL và lý do"):
            st.markdown(
                "- **Lưu lượng truy cập, PageRank, Google index, số liên kết trỏ tới trang, báo cáo thống kê**: "
                "cần dịch vụ bên ngoài (Alexa, PageRank, PhishTank…) đã ngừng hoặc phải trả phí.\n"
                "- **Dịch vụ rút gọn link, chuyển hướng bằng //, thời hạn đăng ký tên miền, favicon, chữ 'https' trong "
                "tên miền, URL bất thường, số lần chuyển hướng**: trong dữ liệu huấn luyện, chiều mã hóa của các đặc "
                "trưng này ngược với quy tắc gốc, nên nếu tự đo theo quy tắc sẽ đưa tín hiệu sai vào mô hình.")
    st.info("Kết quả chỉ để tham khảo: mô hình học từ bộ dữ liệu UCI (thu thập khoảng 2012–2015) với các quy tắc "
            "đặc trưng đơn giản, nên độ chính xác trên website hiện nay có thể thấp hơn số đo trên tập test.")

with tab_test:
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    c1, c2, c3 = st.columns([2, 1, 1])
    picked = c1.number_input(f"Chọn mẫu trong tập test (0 – {len(Xte) - 1})", 0, len(Xte) - 1,
                             value=st.session_state.idx, step=1)
    if c2.button("🎲 Mẫu ngẫu nhiên", use_container_width=True):
        picked = int(np.random.randint(len(Xte)))
    if c3.button("✅ Toàn bộ hợp lệ", use_container_width=True):
        st.session_state.all_ok = True
    if picked != st.session_state.idx:
        st.session_state.idx = picked
        st.session_state.pop("all_ok", None)
        for c in cols:  # nạp lại giá trị của mẫu mới vào các ô
            st.session_state.pop(f"f_{c}", None)
        st.rerun()
    if st.session_state.pop("all_ok", False):
        for c in cols:
            st.session_state[f"f_{c}"] = 1

    row = Xte.iloc[st.session_state.idx]
    result = st.container()  # kết quả hiển thị phía trên các ô đặc trưng
    st.markdown("### Đặc trưng của website")
    st.caption("Giá trị đặc trưng lấy từ mẫu đã chọn; có thể sửa từng ô để thử “nếu… thì…”.")
    grid = st.columns(3)
    x = {}
    for i, c in enumerate(cols):
        opts = sorted(Xte[c].unique().tolist(), reverse=True)
        key = f"f_{c}"
        if key not in st.session_state:
            st.session_state[key] = int(row[c])
        x[c] = grid[i % 3].selectbox(LABELS.get(c, c), opts, key=key, format_func=VALUE_TEXT.get)

    edited = any(x[c] != row[c] for c in cols)
    prob = model.predict_proba(pd.DataFrame([x])[cols].to_numpy())[0, 1]
    with result:
        st.markdown("### Kết quả")
        card(prob, None if edited else yte[st.session_state.idx])
        if edited:
            st.caption("Đã sửa đặc trưng nên không còn nhãn thật để so sánh.")

with tab_perf:
    p = model.predict(Xte.to_numpy())
    k = st.columns(4)
    k[0].metric("F1 (test)", f"{f1_score(yte, p):.4f}")
    k[1].metric("Precision", f"{precision_score(yte, p):.4f}")
    k[2].metric("Recall", f"{recall_score(yte, p):.4f}")
    k[3].metric("Số mẫu test", f"{len(yte)}")
    left, right = st.columns(2)
    cm = confusion_matrix(yte, p)
    left.markdown("**Ma trận nhầm lẫn**")
    left.dataframe(pd.DataFrame(cm, index=["Thật: hợp lệ", "Thật: giả mạo"],
                                columns=["Dự đoán: hợp lệ", "Dự đoán: giả mạo"]), use_container_width=True)
    left.markdown("**Siêu tham số đã chọn**")
    left.dataframe(pd.DataFrame(dec["params"].items(), columns=["Siêu tham số", "Giá trị"]),
                   hide_index=True, use_container_width=True)
    imp = pd.DataFrame({"Đặc trưng": [LABELS.get(c, c) for c in cols], "Mức quan trọng": model.feature_importances_})
    right.markdown("**10 đặc trưng quan trọng nhất**")
    right.dataframe(imp.nlargest(10, "Mức quan trọng"), hide_index=True, use_container_width=True,
                    column_config={"Mức quan trọng": st.column_config.ProgressColumn(
                        format="%.3f", min_value=0.0, max_value=float(imp["Mức quan trọng"].max()))})
