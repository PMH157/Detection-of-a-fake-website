"""Kiểm tra quy tắc trích xuất đặc trưng (không cần mạng). Chạy: pytest web"""
from url_features import html_part, site, url_part


def test_site():
    assert site("mail.google.com") == "google.com"
    assert site("a.b.com.vn") == "b.com.vn"
    assert site("tuoitre.vn") == "tuoitre.vn"


def test_url_part():
    f, _ = url_part("http://192.168.1.10:8080/login@x", "192.168.1.10")
    assert f["having_IP_Address"] == -1 and f["having_At_Symbol"] == -1 and f["port"] == -1
    f, _ = url_part("https://www.google.com/", "www.google.com")
    assert f == {"having_IP_Address": 1, "URL_Length": 1, "having_At_Symbol": 1, "Prefix_Suffix": 1,
                 "having_Sub_Domain": 1, "port": 1}
    f, _ = url_part("http://secure-login.paypal.com.account.verify.example.com/" + "a" * 60, "secure-login.paypal.com.account.verify.example.com")
    assert f["Prefix_Suffix"] == -1 and f["having_Sub_Domain"] == -1 and f["URL_Length"] == -1


def test_html_part():
    good = ('<html><head><script src="/app.js"></script><link href="/s.css"></head><body>'
            '<img src="/a.png"><a href="/about">x</a><a href="https://shop.example.com/">y</a>'
            '<form action="/search"></form></body></html>')
    f, _ = html_part(good, "https://example.com/", "example.com")
    assert all(f[c] == 1 for c in ("Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
                                   "on_mouseover", "RightClick", "popUpWidnow", "Iframe"))
    bad = ('<body oncontextmenu="return false"><img src="https://evil.cc/a.png"><img src="https://evil.cc/b.png">'
           '<a href="#">1</a><a href="javascript:void(0)">2</a><a href="https://evil.cc">3</a>'
           '<script src="https://evil.cc/x.js"></script><form action="about:blank"></form>'
           '<iframe src="https://evil.cc" frameborder="0" width="0" height="0"></iframe></body>')
    f, _ = html_part(bad, "https://example.com/", "example.com")
    assert f["Request_URL"] == -1 and f["URL_of_Anchor"] == -1 and f["Links_in_tags"] == -1
    assert f["SFH"] == -1 and f["RightClick"] == -1 and f["Iframe"] == -1


def test_empty_page_is_not_measured():
    f, _ = html_part("<html><body>blocked</body></html>", "https://example.com/", "example.com")
    assert f["Request_URL"] is None and f["URL_of_Anchor"] is None and f["Links_in_tags"] is None
