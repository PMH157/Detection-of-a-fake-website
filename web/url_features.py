"""Trích xuất đặc trưng UCI Phishing Websites từ một URL thật.

Quy ước giá trị như bộ dữ liệu: 1 = hợp lệ, 0 = nghi ngờ, -1 = dấu hiệu giả mạo.
Chỉ tính những đặc trưng đo được và có chiều mã hóa khớp với dữ liệu huấn luyện (xem USED);
đặc trưng không đo được trả về None và bị loại khỏi mô hình.
"""
import ipaddress
import json
import re
import socket
import ssl
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

# 18 đặc trưng dùng cho chế độ URL. Bị loại: web_traffic, Page_Rank, Google_Index, Links_pointing_to_page,
# Statistical_report (dịch vụ bên ngoài đã ngừng/trả phí) và các đặc trưng có chiều mã hóa trong dữ liệu
# ngược với quy tắc gốc (Shortining_Service, double_slash_redirecting, Domain_registeration_length,
# Favicon, HTTPS_token, Abnormal_URL, Redirect).
USED = ["having_IP_Address", "URL_Length", "having_At_Symbol", "Prefix_Suffix", "having_Sub_Domain",
        "SSLfinal_State", "port", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
        "Submitting_to_email", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord"]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
SLD = {"com", "net", "org", "edu", "gov", "ac", "co", "info", "biz", "name", "int"}


def site(host):
    """Tên miền đăng ký, vd mail.google.com -> google.com, a.b.com.vn -> b.com.vn."""
    # shortcut: danh sách SLD tự viết thay cho Public Suffix List, đủ cho .com.vn/.co.uk; dùng tldextract nếu cần chính xác
    parts = host.lower().strip(".").split(".")
    n = 3 if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in SLD else 2
    return ".".join(parts[-n:])


def ratio_rule(ratio, low, high):
    return 1 if ratio < low else (0 if ratio <= high else -1)


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, {k: (v or "") for k, v in attrs}))


def url_part(url, host):
    """Đặc trưng chỉ cần chuỗi URL."""
    f, why = {}, {}
    try:
        ipaddress.ip_address(host.strip("[]"))
        is_ip = True
    except ValueError:
        is_ip = bool(re.fullmatch(r"(0x[0-9a-f]+\.?)+", host, re.I))
    f["having_IP_Address"], why["having_IP_Address"] = (-1, "tên miền là địa chỉ IP") if is_ip else (1, "dùng tên miền")
    n = len(url)
    f["URL_Length"], why["URL_Length"] = ratio_rule(n, 54, 75), f"{n} ký tự (<54 hợp lệ, 54–75 nghi ngờ)"
    f["having_At_Symbol"], why["having_At_Symbol"] = (-1, "có '@'") if "@" in url else (1, "không có '@'")
    f["Prefix_Suffix"], why["Prefix_Suffix"] = (-1, f"'{host}' có '-'") if "-" in host else (1, "không có '-'")
    h = host[4:] if host.startswith("www.") else host
    h = re.sub(r"\.[a-z]{2}$", "", h)  # bỏ ccTLD như quy tắc gốc
    dots = h.count(".")
    f["having_Sub_Domain"] = None if is_ip else 1 if dots <= 1 else (0 if dots == 2 else -1)  # IP: không có tên miền con
    why["having_Sub_Domain"] = f"{dots} dấu chấm sau khi bỏ www và ccTLD"
    port = urlsplit(url).port
    f["port"], why["port"] = (-1, f"cổng {port}") if port not in (None, 80, 443) else (1, "cổng chuẩn")
    return f, why


def html_part(html, base, base_site):
    """Đặc trưng lấy từ mã HTML của trang."""
    p = Tags()
    p.feed(html)
    ext = lambda u: urlsplit(urljoin(base, u)).hostname not in (None, "") and site(urlsplit(urljoin(base, u)).hostname) != base_site  # noqa: E731
    f, why = {}, {}

    objs = [a.get("src", "") for t, a in p.tags if t in ("img", "audio", "video", "embed", "source") and a.get("src")]
    # trang không có phần tử nào để đếm -> không đo được (None), không mặc định là "hợp lệ"
    r = sum(map(ext, objs)) / len(objs) if objs else 0
    f["Request_URL"] = ratio_rule(r, 0.22, 0.61) if objs else None
    why["Request_URL"] = f"{r:.0%} của {len(objs)} tài nguyên từ tên miền khác"

    hrefs = [a.get("href", "").strip() for t, a in p.tags if t == "a"]
    bad = [h for h in hrefs if h in ("", "#", "#content", "#skip") or h.lower().startswith("javascript:") or ext(h)]
    r = len(bad) / len(hrefs) if hrefs else 0
    f["URL_of_Anchor"] = ratio_rule(r, 0.31, 0.67) if hrefs else None
    why["URL_of_Anchor"] = f"{r:.0%} của {len(hrefs)} liên kết <a> trỏ ra ngoài/rỗng"

    links = ([a["src"] for t, a in p.tags if t == "script" and a.get("src")]
             + [a["href"] for t, a in p.tags if t == "link" and a.get("href")]
             + [a["content"] for t, a in p.tags if t == "meta" and a.get("content", "").startswith("http")])
    r = sum(map(ext, links)) / len(links) if links else 0
    f["Links_in_tags"] = ratio_rule(r, 0.17, 0.81) if links else None
    why["Links_in_tags"] = f"{r:.0%} của {len(links)} liên kết meta/script/link ra ngoài"

    actions = [a.get("action", "").strip() for t, a in p.tags if t == "form"]
    sfh = [(-1 if x.lower() in ("", "about:blank") else 0 if ext(x) else 1) for x in actions]
    f["SFH"] = min(sfh) if sfh else 1
    why["SFH"] = f"{len(actions)} form" + ("" if sfh else " (không có form)")
    mail = any(x.lower().startswith("mailto:") for x in actions) or bool(re.search(r"\bmail\s*\(", html))
    f["Submitting_to_email"], why["Submitting_to_email"] = (-1, "form gửi qua email") if mail else (1, "không gửi qua email")

    low = html.lower()
    mo = bool(re.search(r"onmouseover[^>]{0,200}window\.status", low))
    f["on_mouseover"], why["on_mouseover"] = (-1, "đổi window.status") if mo else (1, "không đổi thanh trạng thái")
    rc = bool(re.search(r"event\.button\s*==+\s*2|oncontextmenu\s*=\s*[\"']?\s*return\s+false", low))
    f["RightClick"], why["RightClick"] = (-1, "chặn chuột phải") if rc else (1, "không chặn chuột phải")
    pop = bool(re.search(r"\bprompt\s*\(", low))
    f["popUpWidnow"], why["popUpWidnow"] = (-1, "có hộp nhập pop-up") if pop else (1, "không có pop-up nhập liệu")
    hidden = [a for t, a in p.tags if t == "iframe" and a.get("frameborder", "") == "0"
              and (a.get("width") == "0" or a.get("height") == "0" or re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", a.get("style", "")))]
    f["Iframe"], why["Iframe"] = (-1, f"{len(hidden)} iframe ẩn") if hidden else (1, "không có iframe ẩn")
    return f, why


def fetch(url, timeout=10):
    """Tải trang; trả (html, final_url, ssl_ok). ssl_ok=None khi kết nối không phải https."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        ssl_ok = True
    except urllib.error.HTTPError as e:
        if e.code >= 500:  # lỗi máy chủ/cổng proxy: không phải nội dung của trang
            raise
        r, ssl_ok = e, True  # 403/404/...: trang vẫn trả HTML để phân tích
    except urllib.error.URLError as e:
        if not isinstance(e.reason, ssl.SSLError):
            raise
        r = urllib.request.urlopen(req, timeout=timeout, context=ssl._create_unverified_context())  # vẫn lấy HTML để phân tích
        ssl_ok = False
    final = r.geturl()
    html = r.read(3_000_000).decode(r.headers.get_content_charset() or "utf-8", "replace")
    return html, final, (ssl_ok if final.startswith("https://") else None)


def domain_age_days(domain, timeout=10):
    """Tuổi tên miền theo RDAP (rdap.org); None khi không tra được (vd một số ccTLD như .vn)."""
    try:
        data = json.load(urllib.request.urlopen(f"https://rdap.org/domain/{domain}", timeout=timeout))
    except Exception:
        return None
    for e in data.get("events", []):
        if e.get("eventAction") == "registration":
            d = datetime.fromisoformat(e["eventDate"].replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - d).days
    return None


def extract(url):
    """Trả (features, why, info). features[c] = None nghĩa là không đo được."""
    url = url.strip()
    if not re.match(r"https?://", url, re.I):
        url = "http://" + url
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError("URL không hợp lệ")
    host = parts.hostname.lower()
    f, why = url_part(url, host)
    f.update({c: None for c in USED if c not in f})
    info = {"url": url, "site": site(host)}

    try:
        socket.getaddrinfo(host, None)
        f["DNSRecord"], why["DNSRecord"] = 1, "có bản ghi DNS"
    except socket.gaierror:
        f["DNSRecord"], why["DNSRecord"] = -1, "không tìm thấy bản ghi DNS"

    try:
        if f["DNSRecord"] == -1:
            raise OSError("tên miền không có bản ghi DNS")
        html, final, ssl_ok = fetch(url)
        info["final_url"] = final
        f["SSLfinal_State"] = 1 if ssl_ok else (0 if ssl_ok is False else -1)
        why["SSLfinal_State"] = {1: "https, chứng chỉ hợp lệ", 0: "https nhưng chứng chỉ không hợp lệ",
                                 -1: "không dùng https"}[f["SSLfinal_State"]]
        hf, hw = html_part(html, final, site(urlsplit(final).hostname or host))
        f.update(hf)
        why.update(hw)
    except Exception as e:  # trang không tải được: các đặc trưng HTML/SSL bị loại
        info["fetch_error"] = f"{type(e).__name__}: {e}"

    if not f["having_IP_Address"] == -1:
        age = domain_age_days(info["site"])
        if age is not None:
            f["age_of_domain"], why["age_of_domain"] = (1 if age >= 180 else -1), f"{age} ngày (≥ 6 tháng hợp lệ)"
    return f, why, info
