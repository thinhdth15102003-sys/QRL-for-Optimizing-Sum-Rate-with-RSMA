"""
Claude autonomous monitor — runs while user is off (2026-06-05 special exception).
Polls every 5 min:
  1. When r18/ep_00400 exists → probe gap-decomp final → send Tele verdict → SIGINT r18 → launch r14 ramp 2 in background (ONE-TIME autonomous training launch)
  2. Newest Case 1 (K=5, M=1) run hits ep_01000 → run log-reading analysis → send Tele

Max runtime: 24h. Heartbeat every 4h.
Logs to /tmp/claude_monitor.log
"""
import os, sys, time, json, subprocess, glob
from pathlib import Path

ROOT = Path("/mnt/c/Project/IRS-assisted RSMA Quantum-RL")
PYBIN = "/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11"
TELE = [PYBIN, "comms/tele.py"]
POLL_INTERVAL = 300   # 5 min
HEARTBEAT_INTERVAL = 4 * 3600   # 4h
MAX_RUNTIME = 24 * 3600   # 24h
R18_EP400_CKPT = ROOT / "results/result_18/checkpoints/ep_00400"

def log(msg):
    with open("/tmp/claude_monitor.log", "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

def tele(msg):
    """Fire-and-forget Telegram send."""
    try:
        subprocess.run(TELE + [msg[:3900]], cwd=ROOT, timeout=30, capture_output=True)
        log(f"TELE: {msg[:120]}")
    except Exception as e:
        log(f"TELE FAIL: {e}")

def run_r18_final_probe():
    """Probe r18 trend across all available ckpts. Returns (success, parsed_verdict_text)."""
    log("Running r18 final gap-decomp probe...")
    cmd = [PYBIN, "analysis/probe_assignment_decomp.py",
           "--ckpts",
           "results/result_11/checkpoints/ep_01000",
           "results/result_18/checkpoints/ep_00000",
           "results/result_18/checkpoints/ep_00100",
           "results/result_18/checkpoints/ep_00200",
           "results/result_18/checkpoints/ep_00400",
           "--episodes", "6", "--steps", "15",
           "--out", "results/result_18/decomp_final.txt"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return False, f"probe failed rc={r.returncode}: {r.stderr[-400:]}"
        out = ROOT / "results/result_18/decomp_final.txt"
        if not out.exists():
            return False, "probe completed but output missing"
        report = out.read_text()
        # Extract ASSIGNMENT-QUALITY TRACK section
        lines = report.split('\n')
        in_track = False
        track_lines = []
        for ln in lines:
            if 'ASSIGNMENT-QUALITY TRACK' in ln:
                in_track = True
            if in_track:
                track_lines.append(ln)
                if 'Interpretation key' in ln:
                    break
        track_text = "\n".join(track_lines[:8])
        return True, track_text
    except Exception as e:
        return False, f"probe exception: {e}"

def kill_r18():
    """SIGINT r18 process to flush ckpts cleanly."""
    log("Sending SIGINT to r18 process...")
    subprocess.run(["pkill", "-SIGINT", "-f", "train.py.*phase-warmup.*freeze-phase"],
                   capture_output=True)
    time.sleep(25)  # let train.py flush

def launch_r14_continuation():
    """Launch r14 ramp 2 continuation from ep_01100 in background."""
    log("Launching r14 ramp 2 continuation in background...")
    log_file = open("/tmp/r14_continue.log", "w")
    proc = subprocess.Popen(
        [PYBIN, "train.py",
         "--R-LoS", "0.3",
         "--resume", "results/result_14/checkpoints/ep_01100/agents",
         "--episodes", "800", "--seed", "0"],
        cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
        start_new_session=True
    )
    log(f"r14 launched PID={proc.pid}, log /tmp/r14_continue.log")
    return proc.pid

def find_newest_case1_run():
    """Return (path, hp) of newest dir with K=5, M=1, or None."""
    candidates = []
    for d in sorted((ROOT / "results").iterdir(), reverse=True):
        if not d.name.startswith("result_"):
            continue
        hp_path = d / "hyperparameters.json"
        if not hp_path.exists():
            continue
        try:
            hp = json.loads(hp_path.read_text())
            sys_c = hp.get('system', {})
            if sys_c.get('K') == 5 and sys_c.get('M') == 1:
                candidates.append((d, hp, d.stat().st_mtime))
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[2])  # newest first
    return candidates[0][0], candidates[0][1]

def case1_has_ep1000(case1_dir):
    return (case1_dir / "checkpoints" / "ep_01000").exists()

def analyze_case1_ep1000(case1_dir, hp):
    """Run log-reading skill-style assessment + compare with Training-Case 1/result_1."""
    log("Analyzing Case 1 @ep_01000...")
    log_path = case1_dir / "training_log.txt"
    if not log_path.exists():
        return "log missing"

    txt = log_path.read_text(errors='ignore')
    lines = txt.split('\n')

    # Get latest rolling-50 + critic explVar
    rolling = [l for l in lines if 'rolling-50: reward' in l]
    last_rolling = rolling[-1] if rolling else "(no rolling)"

    # Best reward (latest '*' line)
    stars = [l for l in lines if l.startswith('*') and 'best' not in l.lower()]
    best_line = stars[-1] if stars else "(no star)"

    # Latest critic explVar
    expl_lines = [l for l in lines if 'explVar=' in l]
    last_expl = expl_lines[-1] if expl_lines else ""

    # Reference: Training-Case 1 / result_1
    ref_path = ROOT / "results/Training-Case 1/result_1/training_log.txt"
    ref_at_1000 = "N/A"
    if ref_path.exists():
        ref_txt = ref_path.read_text(errors='ignore')
        ref_rolling = [l for l in ref_txt.split('\n') if 'rolling-50: reward' in l]
        # rolling printed every 12 ep, so ep~1000 ≈ line idx ~83
        if len(ref_rolling) >= 83:
            ref_at_1000 = ref_rolling[83]

    lam_d = hp.get('system', {}).get('lambda_D', '?')
    rlos = hp.get('system', {}).get('R_LoS_km', '?')

    msg = (f"📊 Case 1 ({case1_dir.name}) @ep_01000 analysis\n"
           f"  config: K=5 M=1 R_LoS={rlos} λ_D={lam_d}\n"
           f"  current: {last_rolling.strip()}\n"
           f"  best: {best_line.strip()[:130]}\n"
           f"  critic: {last_expl.strip()[:80]}\n"
           f"  ── vs result_1 (λ=2) @ep~1000 ref:\n"
           f"  {ref_at_1000.strip()[:160]}\n"
           f"  (Priority: critic stable first → PhaseMLP → QoS → R_tot per Common-Knowledge P5#8)")
    return msg

def heartbeat(state):
    return (f"💓 Claude monitor alive. "
            f"r18_done={state['r18_done']} case1_analyzed={state['case1_analyzed']} "
            f"runtime={(time.time()-state['start'])/3600:.1f}h")

def main():
    state = {'r18_done': False, 'case1_analyzed': False, 'last_heartbeat': 0, 'start': time.time(),
             'case1_seen_dir': None}
    tele("🤖 Claude autonomous monitor START. Poll 5min, max 24h. Will: (1) r18@ep400 probe→Tele→kill→launch r14 ramp2; (2) Case1 re-run@ep1000 analysis→Tele.")
    log("Monitor started")

    while time.time() - state['start'] < MAX_RUNTIME:
        # --- r18 ep_00400 trigger ---
        if not state['r18_done'] and R18_EP400_CKPT.exists():
            log("r18 ep_00400 ckpt detected!")
            ok, verdict = run_r18_final_probe()
            tele(f"🎯 r18 EXP-3c FINAL @ep_00400 verdict (live, ceiling, gap):\n\n{verdict}\n\n→ Full: results/result_18/decomp_final.txt")
            if ok:
                kill_r18()
                tele("⛔ r18 killed (verdict conclusive at ep_00400, ep_00600 marginal info). Launching r14 ramp 2 continuation from ep_01100...")
                pid = launch_r14_continuation()
                tele(f"🚀 r14 ramp 2 continue launched PID={pid}. log /tmp/r14_continue.log. Target ep_01100→1900 (800 more eps, ~6-12h).")
            state['r18_done'] = True

        # --- Case 1 re-run ep_01000 trigger ---
        if not state['case1_analyzed']:
            found = find_newest_case1_run()
            if found:
                case1_dir, hp = found
                # Skip if same dir as result_3 already at ep_00100 (user said they'll re-launch with λ=2)
                lam_d = hp.get('system', {}).get('lambda_D', None)
                if lam_d != 1.5:  # not the historical λ=1.5 launch, must be new λ=2 run
                    if case1_has_ep1000(case1_dir):
                        log(f"Case 1 {case1_dir.name} hit ep_01000 (λ_D={lam_d})")
                        msg = analyze_case1_ep1000(case1_dir, hp)
                        tele(msg)
                        state['case1_analyzed'] = True
                        state['case1_seen_dir'] = case1_dir.name

        # --- Heartbeat ---
        if time.time() - state['last_heartbeat'] > HEARTBEAT_INTERVAL:
            tele(heartbeat(state))
            state['last_heartbeat'] = time.time()

        # --- Done? ---
        if state['r18_done'] and state['case1_analyzed']:
            tele(f"✓ All tasks complete. Total runtime {(time.time()-state['start'])/3600:.1f}h. Monitor EXIT.")
            log("All done, exiting")
            return 0

        time.sleep(POLL_INTERVAL)

    # Timeout
    tele(f"⏱ Monitor TIMEOUT (24h). r18_done={state['r18_done']} case1_analyzed={state['case1_analyzed']}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
