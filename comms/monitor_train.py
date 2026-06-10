"""
monitor_train.py — Giám sát + ĐIỀU KHIỂN training từ xa qua Telegram (phone/laptop).
READ + CONTROL: đọc log + (tùy lệnh) kill/rerun training. Khóa theo CHAT_ID (chỉ bạn).

PUSH (tự động):
  • status định kỳ, ⚠ cảnh báo TREO (log đứng), 🔴 báo DỪNG.
LỆNH (bạn nhắn bot từ điện thoại):
  /status  ep/QoS/explVar/gradV   /tail [N]  N dòng log cuối (mặc định 20)
  /diag    critic health          /process   list training đang chạy (pid/runtime/args) + GPU
  /kill [result_N|pid]  kill all hoặc 1 run  /rerun  chạy lại training (chặn nếu đang chạy)
  /ramp <R_LoS> <λ> <result_N|ckpt> [eps] [shape lag ent= dnn]  — resume + levers
  /analysis <result_N>  spawn Claude session phân tích    /help  danh sách lệnh

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
HANG_MIN    = 12     # log đứng quá N phút → cảnh báo treo
# Auto-push (2026-06-01): notify on EVERY new diag panel (≈ every 12 ep PPO update),
# replacing fixed-interval 30-min status. STATUS_MIN giữ làm fallback nếu diag không
# fire trong N phút (warmup edge case).
STATUS_MIN  = 30     # fallback push (chỉ dùng nếu không có diag mới)
# Lệnh /rerun dùng để chạy lại training (sửa env/episodes nếu cần):
TRAIN_CMD   = (f'cd "{PROJECT_DIR}" && '
               '(source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || '
               ' source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null); '
               'conda activate IRS_QRL; '
               'nohup python train.py --episodes 4000 > /dev/null 2>&1 &')
# Lệnh spawn 1 Claude Code HEADLESS cho /analysis. {prompt} = đã shlex.quote.
# ⚠ Sửa cho đúng máy: cần `claude` CLI trong PATH WSL + đã auth. --dangerously-skip-permissions
#   để chạy autonomous (đọc log + probe + tele.py) không cần người duyệt.
CLAUDE_CMD = 'claude --dangerously-skip-permissions -p {prompt}'
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
    # match ' train.py' or '/train.py' → excludes this bot (monitor_train.py)
    return bool(sh("ps -eo cmd | grep '[ /][t]rain.py'").strip())


def read(log):
    try: return open(log, encoding="utf-8", errors="replace").read()
    except Exception: return ""


def parse(log) -> str:
    """Compact status — per-diag notify (Common-Knowledge Part 5 #8 priority order)."""
    txt = read(log); name = os.path.basename(os.path.dirname(log))
    # episode: prefer latest diag header; fall back to latest episode row
    diag_eps = re.findall(r"diag\[(\d+)\]", txt)
    ep = diag_eps[-1] if diag_eps else (re.findall(r"^\s*\*?\s*(\d+)\s+-?\d", txt, re.M) or ["?"])[-1]
    spe = re.findall(r"([\d.]+)s\s*$", txt, re.M)
    spe_str = f" · {spe[-1]}s/ep" if spe else ""

    # rolling-50 (task: QoS, reward, Rtot, best)
    roll = re.findall(
        r"rolling-\d+: reward μ=\s*([-\d.]+)\s+QoS μ=\s*(\d+)%\s+Rtot μ=([\d.]+)\s*│\s*best=([-\d.]+)\s*\((\d+)\s*ep ago\)",
        txt)
    qos_line = "QoS — · R — · Rtot — · best —"
    if roll:
        r, q, rt, best, ago = roll[-1]
        qos_line = f"QoS {q}% · R {r} · Rtot {rt} · best {best}({ago}ep)"

    # priority 1: critic
    ev = re.findall(r"explVar=([+-][\d.]+)", txt)
    sv = re.findall(r"σV=([\d.]+)", txt)
    crit_line = ""
    if ev:
        crit_line = f"critic: explVar {ev[-1]}"
        if sv:
            crit_line += f" · σV {sv[-1]}"

    # priority 3 quantum: λ frozen [E] + λgrad r (G2 monitor)
    lam = re.findall(r"λ\|max\|\s+y=([\d.]+)\s+z=([\d.]+)", txt)
    lg  = re.findall(r"λgrad:\s+mag=([\d.e+\-]+)\s+net=([\d.e+\-]+)\s+r=([\d.]+)", txt)
    q_parts = []
    if lam:
        q_parts.append(f"λ y={lam[-1][0]} z={lam[-1][1]}")
    if lg:
        r_val = float(lg[-1][2])
        flag = " [frozen]" if r_val < 0.2 else " [moving]"
        q_parts.append(f"λgrad r={lg[-1][2]}{flag}")
    qline = "quantum: " + " · ".join(q_parts) if q_parts else ""

    # priority 2: PhaseMLP idle (ent ph absolute; max ~66 for M=2 N=24 → %max)
    ent = re.findall(r"ent q=[\d.]+\s+ph=([\d.]+)\s+pw=[\d.]+\s+ck=[\d.]+", txt)
    phline = ""
    if ent:
        ph_val = float(ent[-1])
        ph_pct = int(round(ph_val / 66.0 * 100))   # rough %max for Case 2 (M=2, N=24, L=4)
        flag = " [idle]" if ph_pct > 55 else ""
        phline = f"phase: ent ph={ent[-1]} (~{ph_pct}%max){flag}"

    # priority 3: PowerMLP factored (pw axis stats + π_split)
    pw = re.findall(
        r"pw axis \(H/Hmax\): split=(\d+)% common=(\d+)% private=(\d+)%\s*│\s*π_split\(common\)=(\d+)%", txt)
    pwline = ""
    if pw:
        s, c, p, pi = pw[-1]
        pwline = f"power: split={s}% common={c}% private={p}% · π_c={pi}%"

    # flag line: ENTROPY-DOMINATED + other warnings
    edpg = re.findall(r"ent/pg \(β·H/\|L_pg\|\): q=([\d.]+) ph=([\d.]+) pw=([\d.]+) ck=([\d.]+)", txt)
    flags = []
    if edpg:
        q, ph, pw_v, ck = (float(x) for x in edpg[-1])
        doms = [n for n, v in (("q", q), ("ph", ph), ("pw", pw_v), ("ck", ck)) if v > 1.0]
        if doms:
            flags.append("⚠ENT-DOM " + "/".join(doms))
    flag_line = " · ".join(flags) if flags else ""

    parts = [f"📊 {name} · ep {ep}{spe_str}", qos_line]
    for x in (crit_line, qline, phline, pwline, flag_line):
        if x:
            parts.append(x)
    return "\n".join(parts)


def latest_diag_ep(log):
    """Return the latest diag[N] episode number found in log, or None."""
    try:
        eps = re.findall(r"diag\[(\d+)\]", read(log))
        return int(eps[-1]) if eps else None
    except Exception:
        return None


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


def cmd_process():
    """List các training train.py đang chạy: pid · runtime · args chính (R_LoS, eps,
    resume-source, levers) + active output dirs (training_log advancing) + GPU mem."""
    raw = sh("ps -eo pid,etimes,cmd | grep '[ /][t]rain.py'")
    rows = []
    for l in raw.splitlines():
        if not l.strip():
            continue
        parts = l.split(None, 2)
        if len(parts) < 3:
            continue
        pid, et, cmd = parts
        try:    et_h = f"{int(et)//3600}h{(int(et)%3600)//60:02d}m"
        except Exception: et_h = f"{et}s"
        def grab(flag):
            m = re.search(rf"{re.escape(flag)}\s+(\S+)", cmd)
            return m.group(1) if m else None
        rlos = grab("--R-LoS") or "?"
        eps  = grab("--episodes") or "?"
        res  = grab("--resume")
        m    = re.search(r"result_\d+", res) if res else None
        resn = m.group(0) if m else "fresh"
        tags = []
        if "--lagrangian"     in cmd: tags.append("lag")
        if "--routing-shape"  in cmd: tags.append("shape")
        if "--phase-warmup"   in cmd: tags.append("warmup")
        if "--freeze-phase"   in cmd: tags.append("freeze")
        if "--actor-mode classical" in cmd: tags.append("dnn")
        rows.append(f"• pid {pid} · {et_h} · R_LoS={rlos} eps={eps} ←{resn}"
                    + (f" [{' '.join(tags)}]" if tags else ""))
    head = (f"🏃 {len(rows)} training đang chạy:\n" + "\n".join(rows)) if rows \
           else "ℹ️ Không có training nào đang chạy."
    # active OUTPUT dirs (training_log advancing < HANG_MIN)
    now, act = time.time(), []
    for lg in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*", "training_log.txt"))):
        try:
            if (now - os.path.getmtime(lg)) / 60.0 > HANG_MIN:
                continue
        except OSError:
            continue
        rn = os.path.basename(os.path.dirname(lg))
        ep = latest_diag_ep(lg)
        act.append(f"{rn}@ep{ep}" if ep is not None else rn)
    act_line = ("\n📂 đang ghi: " + " · ".join(act)) if act else ""
    g = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
    return f"{head}{act_line}\n🖥️ GPU mem: {g}"


def cmd_kill(arg: str = ""):
    """Không arg → kill TẤT CẢ train.py. arg = <pid> → kill đúng pid đó. arg = <result_N>
    (hoặc bất kỳ chuỗi) → kill train.py có cmd khớp (vd resume-source result_12). Dùng
    /process xem pid trước."""
    arg = arg.strip()
    if not train_running():
        return "ℹ️ Không có training nào đang chạy."
    if not arg:
        sh("for p in $(ps -eo pid,cmd | grep '[ /][t]rain.py' | awk '{print $1}'); do kill -9 $p; done")
        time.sleep(2)
        return ("✅ Đã kill TẤT CẢ training.\n" + cmd_process()) if not train_running() \
               else "⚠️ Vẫn còn process (thử lại /kill)."
    if arg.isdigit():
        chk = sh(f"ps -p {arg} -o cmd= 2>/dev/null")
        if "train.py" not in chk or "monitor_train.py" in chk:
            return f"⚠️ pid {arg} không phải training train.py (hoặc đã tắt). /process để xem."
        sh(f"kill -9 {arg}"); time.sleep(2)
        return f"✅ Đã kill pid {arg}.\n" + cmd_process()
    pids = sh(f"ps -eo pid,cmd | grep '[ /][t]rain.py' | grep -F -- {shlex.quote(arg)} "
              "| awk '{print $1}'").split()
    if not pids:
        return (f"⚠️ Không thấy train.py nào khớp '{arg}'. "
                "Lưu ý: khớp theo cmd (thường là resume-source). /process xem pid rồi /kill <pid>.")
    for p in pids:
        sh(f"kill -9 {p}")
    time.sleep(2)
    return f"✅ Đã kill {len(pids)} process khớp '{arg}' (pid {' '.join(pids)}).\n" + cmd_process()


def cmd_rerun():
    if train_running(): return "⚠️ Training ĐANG chạy → /kill trước (tránh 2 job/GPU deadlock)."
    sh(TRAIN_CMD); time.sleep(4)
    return "✅ Đã khởi chạy training lại." if train_running() else "⚠️ Chưa thấy process — xem TRAIN_CMD/env."


def cmd_ramp(args_str: str) -> str:
    """/ramp <R_LoS> <λ> <result_N|ckpt> [eps] [opts...] — resume sang ramp mới + levers.
    OPTS (sau eps, thứ tự bất kỳ):
      shape[=coef]  → --routing-shape [--routing-shape-coef coef]   (mặc định 0.1; IRS≥Blk lever)
      lag[=target]  → --lagrangian --qos-target target (λ = lambda-min) thay vì λ_D FIXED (né [K])
      ent[=end]     → --beta-entropy-anneal-end end    (mặc định 0.3; entropy cao hơn)
      lmax=X        → --lambda-max X   (chỉ khi lag; mặc định 5.0)
      dnn           → --actor-mode classical (DNN baseline)
    Multi-run OK (chỉ cảnh báo nếu đã có run)."""
    a = args_str.split()
    if len(a) < 3:
        return ("Dùng: /ramp <R_LoS> <λ> <result_N|ckpt> [eps] [shape lag ent= dnn]\n"
                "vd: /ramp 0.5 2.5 result_12 1500 shape lag=0.90   (shaping + Lagrangian)\n"
                "    /ramp 0.4 3 result_8                          (λ FIXED, auto-pick ckpt)\n"
                "    /ramp 0.2 1.5 result_5 2000 shape=0.2 ent=0.3")
    try:
        rlos, lam = float(a[0]), float(a[1])
    except ValueError:
        return "⚠️ R_LoS và λ phải là số (vd: /ramp 0.5 2.5 result_12 shape lag)."
    src = a[2]
    # ── parse trailing opts (eps = số trần đầu tiên; còn lại = key[=val]) ──
    eps = 1500
    use_lag, qos_t, lag_max = False, 0.90, 5.0
    shape, shape_coef = False, 0.1
    ent_end, dnn = None, False
    for tok in a[3:]:
        if tok.isdigit():
            eps = int(tok); continue
        k, _, v = tok.partition("=")
        k = k.lower()
        try:
            if   k in ("shape", "routing"): shape = True;  shape_coef = float(v) if v else 0.1
            elif k in ("lag", "lagrangian"): use_lag = True; qos_t = float(v) if v else 0.90
            elif k in ("ent", "entropy"):   ent_end = float(v) if v else 0.3
            elif k in ("lmax", "lambdamax"): lag_max = float(v) if v else 5.0
            elif k in ("dnn", "classical"): dnn = True
            # else: bỏ qua token lạ
        except ValueError:
            return f"⚠️ Giá trị không hợp lệ ở '{tok}'."
    # ── resolve resume target ──
    if ("checkpoints" in src) or ("ep_" in src):
        cdir = src if src.startswith("/") else os.path.join(RESULTS_DIR, src)
        if not cdir.rstrip("/").endswith("agents"):
            cdir = os.path.join(cdir, "agents")
        if not os.path.isdir(cdir):
            return f"⚠️ Ckpt không tồn tại: {cdir}"
        pick_note = ""
    else:
        cdir = src if src.startswith("/") else os.path.join(RESULTS_DIR, src)
        if not os.path.isdir(os.path.join(cdir, "checkpoints")):
            return f"⚠️ Không thấy {src}/checkpoints/."
        pick_note = " (train.py auto-pick BEST ckpt)"
    # ── build flags ──
    if use_lag:
        lam_part = (f"--lagrangian --qos-target {qos_t} --lambda-lr 0.10 "
                    f"--lambda-min {lam} --lambda-max {lag_max}")
        lam_desc = f"Lagrangian target={qos_t} λ∈[{lam},{lag_max}]"
        krisk = ""
    else:
        lam_part = f"--lambda-D {lam}"
        lam_desc = f"λ_D={lam} FIXED"
        krisk = "⚠ λ FIXED bump trên resume = [K] risk → /status theo dõi Ck/explVar 50-100ep.\n"
    opts = []
    if shape:               opts.append(f"--routing-shape --routing-shape-coef {shape_coef}")
    if ent_end is not None:  opts.append(f"--beta-entropy-anneal-end {ent_end}")
    if dnn:                  opts.append("--actor-mode classical")
    opt_str = " ".join(opts)

    def nproc():
        try: return int(sh("ps -eo cmd | grep -c '[t]rain.py'").strip())
        except Exception: return -1
    n0 = nproc()
    cmd = (f'cd "{PROJECT_DIR}" && '
           '(source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || '
           ' source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null); '
           'conda activate IRS_QRL; '
           f'nohup python train.py --R-LoS {rlos} {lam_part} {opt_str} '
           f'--resume "{cdir}" --episodes {eps} > /dev/null 2>&1 &')
    sh(cmd); time.sleep(5)
    n1 = nproc()
    par = f"⚠️ {n0} training khác đang chạy (GPU parallel — coi chừng OOM).\n" if n0 > 0 else ""
    status = (f"✅ process mới đã lên (train.py: {n0}→{n1})." if n1 > n0
              else f"⚠️ chưa thấy process mới (kiểm tra env/path). train.py: {n0}→{n1}")
    return (f"{par}🚀 RAMP: R_LoS={rlos} · {lam_desc} · eps={eps}\n"
            f"levers: {opt_str or '(none)'}\nresume: {cdir}{pick_note}\n{status}\n{krisk}")


def cmd_analysis(args_str: str) -> str:
    """/analysis <result_N> — TỰ TẠO tmux session Claude MỚI, chạy analysis run đó, báo về Telegram.
    Spawn session RIÊNG (không đụng session chính nếu đang mở)."""
    a = args_str.strip().split()
    if not a:
        return "Dùng: /analysis <result_N>   (vd: /analysis result_8)"
    res = a[0]
    run_dir = res if res.startswith("/") else os.path.join(RESULTS_DIR, res)
    if not os.path.isdir(run_dir):
        return f"⚠️ Không thấy run-dir: {run_dir}"
    if "TMUXOK" not in sh("command -v tmux >/dev/null && echo TMUXOK"):
        return "⚠️ tmux không có trong WSL (sudo apt install tmux)."
    if "CLOK" not in sh("command -v claude >/dev/null && echo CLOK"):
        return ("⚠️ 'claude' CLI không thấy trong PATH WSL → không spawn được. "
                "Cài + auth Claude Code trong WSL.")
    rn = os.path.basename(run_dir.rstrip("/"))
    name = f"ana_{rn}_{int(time.time()) % 100000}"
    prompt = (f"Analyze training run results/{rn} (IRS-RSMA project) using the irs-rsma-log-reading "
              f"skill: R_tot/QoS trend, critic + per-head health, any [K]/drift/degeneration. "
              f"Run ONE cheap probe only if it changes the verdict. Then send a SHORT (<800 char) "
              f'verdict to Telegram: cd "{PROJECT_DIR}" && python comms/tele.py "...". '
              f"Be concise. Do NOT launch/kill training.")
    claude = CLAUDE_CMD.format(prompt=shlex.quote(prompt))
    inner = (f'cd "{PROJECT_DIR}" && {claude}; '
             f'python comms/tele.py "[analysis {rn}] session kết thúc."')
    sh(f'tmux new-session -d -s {shlex.quote(name)} {shlex.quote(inner)}')
    time.sleep(2)
    ok = "OK" in sh(f'tmux has-session -t {shlex.quote(name)} 2>/dev/null && echo OK')
    return (f"🔬 Spawned analysis session '{name}' cho {rn}.\n"
            f"{'✅ tmux session đang chạy — Claude phân tích + đẩy verdict về đây khi xong.' if ok else '⚠️ session ko thấy — kiểm tra claude CLI/auth/quota.'}\n"
            f"(xem trực tiếp: tmux attach -t {name})")


HELP = ("🤖 Lệnh:\n/status · /tail [N] · /diag · /process · /rerun\n"
        "/kill [result_N|pid] — kill tất cả, hoặc 1 run cụ thể (xem pid ở /process)\n"
        "/ramp <R_LoS> <λ> <result_N|ckpt> [eps] [shape lag ent= dnn] — resume + levers\n"
        "   vd: /ramp 0.5 2.5 result_12 1500 shape lag=0.90\n"
        "/analysis <result_N> — spawn Claude session phân tích run đó + báo về đây\n/help")


def handle(text: str) -> str:
    t = text.strip().split()
    c = t[0].lower()
    log = latest_log()
    if c == "/start" or c == "/help": return HELP
    if log is None and c in ("/status", "/tail", "/diag"): return "(chưa có log)"
    if c == "/status": return parse(log)
    if c == "/tail":   return cmd_tail(log, int(t[1]) if len(t) > 1 and t[1].isdigit() else 20)
    if c == "/diag":   return cmd_diag(log)
    if c in ("/process", "/gpu", "/ps"): return cmd_process()   # /gpu alias giữ cho quen tay
    if c == "/kill":   return cmd_kill(text[len(t[0]):].strip())
    if c == "/rerun":  return cmd_rerun()
    if c == "/ramp":   return cmd_ramp(text[len(t[0]):].strip())
    if c == "/analysis": return cmd_analysis(text[len(t[0]):].strip())
    return "❓ " + HELP


def main():
    if "PASTE_" in BOT_TOKEN: print("Chưa điền token."); return
    send("🟢 control bot online. " + HELP)
    # drain update cũ để không chạy lại lệnh trước khi khởi động
    offset = None
    old = get_updates(None, timeout=0)
    if old: offset = old[-1]["update_id"] + 1
    # MULTI-RUN: state keyed per-run (result_N) → push cho MỌI run song song.
    last_status, hang_alerted, last_diag_ep = {}, {}, {}
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
        # ── push tự động (MULTI-RUN: push cho MỌI result_* đang active song song) ──
        now = time.time()
        for log in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*", "training_log.txt"))):
            run = os.path.basename(os.path.dirname(log))      # "result_N"
            try: stale = (now - os.path.getmtime(log)) / 60.0
            except OSError: continue
            if stale <= HANG_MIN:
                # ACTIVE run → push mỗi diag[N] mới (label theo run), fallback STATUS_MIN.
                hang_alerted[run] = False
                cur = latest_diag_ep(log)
                if cur is not None and cur > last_diag_ep.get(run, -1):
                    send(f"[{run}]\n{parse(log)}"); last_diag_ep[run] = cur; last_status[run] = now
                elif now - last_status.get(run, 0.0) > STATUS_MIN * 60:
                    send(f"[{run}]\n{parse(log)}"); last_status[run] = now
            elif run in last_diag_ep and not hang_alerted.get(run, False):
                # Chỉ alert run TỪNG active trong session này rồi đứng (treo/dừng) → tránh spam run cũ.
                send(f"⚠️ [{run}] log đứng {stale:.0f} phút (treo hoặc đã dừng) — kiểm tra.\n{parse(log)}\n→ /process /kill /rerun")
                hang_alerted[run] = True


if __name__ == "__main__":
    main()
