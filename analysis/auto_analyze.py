"""
auto_analyze.py — RELIABLE background watcher for result_8.

NOT a Claude self-wake (that's unreliable). A plain process that:
  1. polls results/result_8/training_log.txt for current episode,
  2. pushes a Telegram milestone when it passes ep 400 (β-anneal done),
  3. at ep >= TARGET (450) runs analyze_critic_run.py and pushes the
     ★T3 STABLE/WOBBLE verdict + T1/T5 + QoS/reward trajectory to Telegram,
  4. alerts if the run dies early.

Reuses tele.py for sending.  Self-terminates after the verdict or MAX_HOURS.

Run (WSL):
  cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL"
  nohup ~/miniconda3/envs/IRS_QRL/bin/python auto_analyze.py > /dev/null 2>&1 &
"""

# ── path bootstrap: make project root importable when run as script ──────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────

import os, re, time, subprocess

PROJ   = os.path.dirname(os.path.abspath(__file__))
PY     = os.path.join(os.path.expanduser("~"),
                      "miniconda3/envs/IRS_QRL/bin/python")
RUN    = os.path.join(PROJ, "results", "result_8")
LOG    = os.path.join(RUN, "training_log.txt")
SUMM   = os.path.join(RUN, "analysis_critic", "summary.txt")
TARGET = 450
ANNEAL = 400
MAX_HOURS = 6
POLL   = 60


def tele(msg: str):
    try:
        subprocess.run([PY, "tele.py", msg], cwd=PROJ, timeout=30)
    except Exception as e:
        print("tele failed:", e)


def read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""


def cur_ep() -> int:
    m = re.findall(r"^\s*\*?\s*(\d+)\s+-?\d", read(LOG), re.M)
    return int(m[-1]) if m else 0


def train_alive() -> bool:
    try:
        out = subprocess.run(["bash", "-lc", "ps -eo cmd | grep -v grep"],
                             capture_output=True, text=True, timeout=15).stdout
        return "train.py --episodes 4000" in out
    except Exception:
        return True


def qos_traj() -> str:
    rolls = re.findall(r"rolling-\d+: reward μ=\s*([-\d.]+)\s+QoS μ=\s*(\d+)%\s+Rtot μ=([\d.]+)",
                       read(LOG))
    if not rolls:
        return "(no rolling yet)"
    tail = rolls[-6:]
    qs = " ".join(f"{q}%" for (_r, q, _rt) in tail)
    last_r, last_q, last_rt = tail[-1]
    return f"QoS {qs}\nlatest: QoS {last_q}% reward {last_r} Rtot {last_rt}"


def summary_headline() -> str:
    txt = read(SUMM)
    if not txt:
        return "(summary.txt trống)"
    lines = txt.splitlines()
    out, grab = [], False
    for ln in lines:
        if "HEADLINE" in ln:
            grab = True
        if "★ T3" in ln or grab:
            out.append(ln)
        if grab and "frac>0.5" in ln:
            # include the verdict line right after, then stop a couple lines later
            idx = lines.index(ln)
            out += lines[idx + 1: idx + 3]
            break
    return "\n".join(out[:18])[:2500] if out else txt[:1500]


def run_analyzer_and_push():
    try:
        subprocess.run([PY, "analyze_critic_run.py", RUN, "--no-plots"],
                       cwd=PROJ, timeout=300)
    except Exception as e:
        tele(f"🔬 result_8: analyzer lỗi: {e}")
        return
    msg = (f"🔬 result_8 @ep{cur_ep()} — ANALYZER VERDICT (T3 PopArt)\n"
           f"{'-'*30}\n{summary_headline()}\n{'-'*30}\n{qos_traj()}\n"
           f"→ Claude đọc kỹ + quyết bước kế khi bạn quay lại.")
    tele(msg)


def main():
    t0 = time.time()
    tele(f"🔬 auto-analyze ONLINE — canh result_8, sẽ đẩy verdict khi đạt ep{TARGET} "
         f"(báo mốc ep{ANNEAL} anneal). Bạn cứ off.")
    fired = set()
    while time.time() - t0 < MAX_HOURS * 3600:
        ep = cur_ep()
        if not train_alive() and ep < TARGET and ep > 0:
            tele(f"🔴 result_8 DỪNG sớm @ep{ep} (trước {TARGET}).\n{qos_traj()}\n→ kiểm tra.")
            break
        if ep >= ANNEAL and ANNEAL not in fired:
            tele(f"🔬 result_8 qua ep{ANNEAL} (β-anneal xong) · {qos_traj()}\n"
                 f"→ chờ ep{TARGET} cho verdict.")
            fired.add(ANNEAL)
        if ep >= TARGET and TARGET not in fired:
            run_analyzer_and_push()
            fired.add(TARGET)
            break
        time.sleep(POLL)
    else:
        tele("🔬 auto-analyze hết giờ (6h) — tự tắt. Dùng /diag /status xem tiếp.")


if __name__ == "__main__":
    main()
