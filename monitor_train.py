"""
monitor_train.py — Giám sát + ĐIỀU KHIỂN training từ xa qua Telegram (phone/laptop).
READ + CONTROL: đọc log + (tùy lệnh) kill/rerun training. Khóa theo CHAT_ID (chỉ bạn).

PUSH (tự động):
  • status định kỳ, ⚠ cảnh báo TREO (log đứng), 🔴 báo DỪNG.
LỆNH (bạn nhắn bot từ điện thoại):
  /status  ep/QoS/explVar/gradV   /tail [N]  N dòng log cuối (mặc định 20)
  /diag    critic health          /gpu       nvidia-smi (util/mem)
  /kill    kill training (treo)    /rerun     chạy lại training (chặn nếu đang chạy)
  /help    danh sách lệnh

Chạy (WSL, ĐÚNG thư mục project):
  cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL"
  nohup python monitor_train.py > monitor.log 2>&1 &
"""
import os, re, glob, time, json, shlex, subprocess, urllib.parse, urllib.request

# ── CONFIG ──────────────────────────────────────────────────────────────────
try:
    from telegram_secrets import BOT_TOKEN, CHAT_ID   # gitignored local file
except Exception:
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PASTE_TOKEN")
    CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "PASTE_CHAT_ID")
PROJECT_DIR = "/mnt/c/Project/IRS-assisted RSMA Quantum-RL"
RESULTS_DIR = PROJECT_DIR + "/results"
INBOX       = RESULTS_DIR + "/claude_inbox.jsonl"   # /claude messages → file queue (no tmux)
STATUS_MIN  = 30     # gửi status định kỳ mỗi N phút
HANG_MIN    = 12     # log đứng quá N phút → cảnh báo treo
# Lệnh /rerun dùng để chạy lại training (sửa env/episodes nếu cần):
TRAIN_CMD   = (f'cd "{PROJECT_DIR}" && '
               '(source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || '
               ' source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null); '
               'conda activate IRS_QRL; '
               'nohup python train.py --episodes 4000 > /dev/null 2>&1 &')
TMUX_TARGET = "claude"   # tmux session/window đang chạy Claude Code (cho lệnh /claude)
API = f"https://api.telegram.org/bot{BOT_TOKEN}"
# ────────────────────────────────────────────────────────────────────────────


def send(text: str) -> None:
    try:
        data = urllib.parse.urlencode({"chat_id": CHAT_ID, "text": text[:4000]}).encode()
        urllib.request.urlopen(urllib.request.Request(API + "/sendMessage", data=data), timeout=20).read()
    except Exception as e:
        print("send failed:", e)


def get_updates(offset, timeout=25):
    try:
        q = urllib.parse.urlencode({"offset": offset or "", "timeout": timeout})
        r = urllib.request.urlopen(API + "/getUpdates?" + q, timeout=timeout + 10).read()
        return json.loads(r).get("result", [])
    except Exception as e:
        print("getUpdates failed:", e); return []


def sh(cmd: str) -> str:
    try:
        return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"(lỗi: {e})"


def latest_log():
    logs = glob.glob(os.path.join(RESULTS_DIR, "result_*", "training_log.txt"))
    return max(logs, key=os.path.getmtime) if logs else None


def train_running() -> bool:
    return "train.py" in sh("ps -eo cmd | grep -v grep")


def read(log):
    try: return open(log, encoding="utf-8", errors="replace").read()
    except Exception: return ""


def parse(log) -> str:
    txt = read(log); name = os.path.basename(os.path.dirname(log))
    ep  = (re.findall(r"^\s*\*?\s*(\d+)\s+-?\d", txt, re.M) or ["?"])[-1]
    roll = re.findall(r"rolling-\d+: reward μ=\s*([-\d.]+)\s+QoS μ=\s*(\d+)%\s+Rtot μ=([\d.]+)", txt)
    qos  = f"QoS {roll[-1][1]}% · reward {roll[-1][0]} · Rtot {roll[-1][2]}" if roll else "—"
    ev   = re.findall(r"explVar=([+-][\d.]+)", txt)
    gv   = re.findall(r"V=([\d.]+)\s*$", txt, re.M)
    spe  = re.findall(r"([\d.]+)s\s*$", txt, re.M)
    crit = (f"explVar {ev[-1]}" if ev else "") + (f" · ‖∇‖V {gv[-1]}" if gv else "")
    sep  = f" · {spe[-1]}s/ep" if spe else ""
    return f"📊 {name} · ep {ep}\n{qos}\n{crit}{sep}"


def cmd_tail(log, n=20):
    lines = read(log).rstrip().splitlines()[-n:]
    return "📜 " + os.path.basename(os.path.dirname(log)) + f" (cuối {n} dòng):\n" + "\n".join(lines)[:3800]


def cmd_diag(log):
    """Hiện block diag cuối + block crit-diag cuối (Tier-2: T1-T5 PopArt /
    clipfrac / bias / explVar / PopArt μσ / POP|ΔV|). crit-diag là cụm có nhiều
    dòng thụt đầu — bắt theo dòng để lấy đủ."""
    lines = read(log).splitlines()
    def _block(tag):
        starts = [i for i, ln in enumerate(lines) if tag in ln]
        if not starts:
            return ""
        i = starts[-1]
        out = [lines[i]]
        for j in range(i + 1, min(i + 12, len(lines))):
            if lines[j].startswith("            ") or lines[j].lstrip().startswith("·"):
                out.append(lines[j])
            else:
                break
        return "\n".join(out)
    d  = _block("┄ diag[")
    cd = _block("crit-diag[")
    parts = []
    if d:  parts.append("🩺 diag:\n" + d)
    if cd: parts.append("🔬 crit-diag (T1-T5):\n" + cd)
    return ("\n\n".join(parts) or "(chưa có diag)")[:3800]


def cmd_gpu():
    g = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader")
    p = sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    return f"🖥️ GPU: {g}\ncompute-apps:\n{p or '(none)'}"


def cmd_kill():
    if not train_running(): return "ℹ️ Không có training nào đang chạy."
    sh("for p in $(ps -eo pid,cmd | grep train.py | grep -v grep | awk '{print $1}'); do kill -9 $p; done")
    time.sleep(2)
    return ("✅ Đã kill training.\n" + cmd_gpu()) if not train_running() else "⚠️ Vẫn còn process (thử lại /kill)."


def cmd_rerun():
    if train_running(): return "⚠️ Training ĐANG chạy → /kill trước (tránh 2 job/GPU deadlock)."
    sh(TRAIN_CMD); time.sleep(4)
    return "✅ Đã khởi chạy training lại." if train_running() else "⚠️ Chưa thấy process — xem TRAIN_CMD/env."


def cmd_claude(msg: str) -> str:
    """Ghi tin của user vào FILE INBOX (claude_inbox.jsonl) — KHÔNG cần tmux.
    Session Claude đang chạy sẽ tự poll inbox định kỳ, xử lý, rồi đẩy kết quả
    về Telegram (qua tele.py). Không cần mở Claude mới."""
    if not msg.strip():
        return "Dùng: /claude <tin nhắn cho Claude>  (vd: /claude phân tích result_7 hiện tại)"
    # ── HYBRID: thử tmux inject TRƯỚC (real-time nếu Claude chạy trong tmux WSL) ──
    injected = (f'[Lệnh từ điện thoại qua Telegram] {msg} '
                f'— (làm xong gửi kết quả NGẮN GỌN về Telegram: '
                f'cd "{PROJECT_DIR}" && python tele.py "kết quả")')
    has_tmux = "SESSIONOK" in sh(
        f'tmux has-session -t {shlex.quote(TMUX_TARGET)} 2>/dev/null && echo SESSIONOK')
    if has_tmux:
        r = sh(f'tmux send-keys -t {shlex.quote(TMUX_TARGET)} -l -- {shlex.quote(injected)} '
               f'&& tmux send-keys -t {shlex.quote(TMUX_TARGET)} Enter && echo SENTOK')
        if "SENTOK" in r:
            return (f"📨 Đã bơm thẳng vào Claude (tmux '{TMUX_TARGET}') — real-time. "
                    f"Chờ Claude xử lý + đẩy kết quả về đây.")
    # ── FALLBACK: không có tmux → LƯU inbox + trả status ngay ──
    rec = {"ts": time.time(), "msg": msg.strip(), "done": False}
    try:
        with open(INBOX, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log = latest_log()
        snap = parse(log) if log else "(chưa có log)"
        return ("📨 Đã LƯU tin vào hàng đợi (không tmux → không real-time; trả lời khi "
                "bạn quay lại session & ping, xem /inbox).\n"
                "⚡ Cần xem NGAY: /status /diag /tail — trực tiếp:\n\n" + snap)
    except Exception as e:
        return f"⚠️ Lỗi ghi inbox: {e}"


def cmd_inbox() -> str:
    """Hiển thị trạng thái hàng đợi inbox (tin chờ xử lý / đã xong)."""
    if not os.path.exists(INBOX):
        return "📭 Inbox trống (chưa có /claude nào)."
    pend, done = [], 0
    for ln in read(INBOX).splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("done"):
            done += 1
        else:
            pend.append(r.get("msg", "")[:60])
    head = f"📬 Inbox: {len(pend)} chờ · {done} xong"
    if pend:
        head += "\nChờ:\n" + "\n".join(f"• {m}" for m in pend[-5:])
    return head


HELP = ("🤖 Lệnh:\n/status · /tail [N] · /diag · /gpu · /kill · /rerun\n"
        "/claude <tin> — nhắn Claude qua inbox (KHÔNG cần tmux) · /inbox — xem hàng đợi\n/help")


def handle(text: str) -> str:
    t = text.strip().split()
    c = t[0].lower()
    log = latest_log()
    if c == "/start" or c == "/help": return HELP
    if log is None and c in ("/status", "/tail", "/diag"): return "(chưa có log)"
    if c == "/status": return parse(log)
    if c == "/tail":   return cmd_tail(log, int(t[1]) if len(t) > 1 and t[1].isdigit() else 20)
    if c == "/diag":   return cmd_diag(log)
    if c == "/gpu":    return cmd_gpu()
    if c == "/kill":   return cmd_kill()
    if c == "/rerun":  return cmd_rerun()
    if c == "/inbox":  return cmd_inbox()
    if c == "/claude": return cmd_claude(text[len(t[0]):].strip())
    return "❓ " + HELP


def main():
    if "PASTE_" in BOT_TOKEN: print("Chưa điền token."); return
    send("🟢 control bot online. " + HELP)
    # drain update cũ để không chạy lại lệnh trước khi khởi động
    offset = None
    old = get_updates(None, timeout=0)
    if old: offset = old[-1]["update_id"] + 1
    last_status, hang_alerted, stop_alerted = 0.0, False, False
    while True:
        for u in get_updates(offset, timeout=25):
            offset = u["update_id"] + 1
            msg = u.get("message") or u.get("edited_message") or {}
            if str(msg.get("chat", {}).get("id")) != str(CHAT_ID):
                continue   # ⛔ chỉ chủ nhân ra lệnh
            text = msg.get("text", "")
            if text:
                try: send(handle(text))
                except Exception as e: send(f"lỗi xử lý lệnh: {e}")
        # ── push tự động ──
        log = latest_log()
        if log is None: continue
        now = time.time()
        stale = (now - os.path.getmtime(log)) / 60.0
        running = train_running()
        if stale > HANG_MIN and running and not hang_alerted:
            send(f"⚠️ NGHI TREO: log đứng {stale:.0f} phút (process còn sống).\n{parse(log)}\n→ /gpu /kill /rerun")
            hang_alerted = True
        if stale <= HANG_MIN: hang_alerted = False
        if not running and not stop_alerted:
            send(f"🔴 Training DỪNG.\n{parse(log)}\n→ /rerun để chạy lại"); stop_alerted = True
        if running: stop_alerted = False
        if now - last_status > STATUS_MIN * 60 and running and stale <= HANG_MIN:
            send(parse(log)); last_status = now


if __name__ == "__main__":
    main()
