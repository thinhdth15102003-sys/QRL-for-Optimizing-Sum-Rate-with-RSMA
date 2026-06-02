"""
tele.py — gửi 1 tin nhắn về Telegram của bạn. Dùng cho Claude đẩy kết quả về điện thoại.
  python tele.py "nội dung cần gửi"
(Claude: khi nhận '[Lệnh từ điện thoại qua Telegram] ...', làm xong chạy lệnh này để báo kết quả.)
"""
import sys, os, urllib.parse, urllib.request

try:
    from telegram_secrets import BOT_TOKEN, CHAT_ID   # gitignored local file
except Exception:
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

def main():
    text = " ".join(sys.argv[1:]).strip() or "(trống)"
    data = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text[:4000]}).encode()
    try:
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data=data, timeout=20).read()
        print("✓ sent to Telegram")
    except Exception as e:
        print("✗ send failed:", e)

if __name__ == "__main__":
    main()
