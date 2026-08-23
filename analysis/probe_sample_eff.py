"""
probe_sample_eff.py
-------------------
Sample-efficiency (A) learning curve + (D) threshold-reaching, VQC vs. DNN at the
R_LoS=0.2 base ramp (Case-2). Pulls per-episode QoS & R_tot from the training logs
(VQC r90 extra >2000-ep segment trimmed; DNN r95). Saves a 2-panel learning-curve PNG
and prints the threshold-reaching (D) numbers.

Usage:
  python analysis/probe_sample_eff.py --vqc results/result_90 --dnn results/result_95 \
     --cap-vqc 2000 --K 10 --out analysis/data/sample_eff_c2.png
"""
import sys, os, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROW = re.compile(r'^\s+(\d+)\s+-?\d')


def parse_log(path, K, cap=None):
    eps, qos, rtot = [], [], []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not ROW.match(line):
                continue
            p = line.split()
            try:                       # Ep Rew Best (—x6) Rtot R/user QoS/K IRS/K Blk/K s/ep
                ep = int(p[0]); rt = float(p[9]); q = float(p[11].split('/')[0]) / K
            except (ValueError, IndexError):
                continue
            eps.append(ep); rtot.append(rt); qos.append(q)
    eps, qos, rtot = np.array(eps), np.array(qos, float), np.array(rtot, float)
    if cap is not None:
        m = eps <= cap; eps, qos, rtot = eps[m], qos[m], rtot[m]
    return eps, qos, rtot


def smooth(y, w=25):
    if len(y) < w:
        return y
    yp = np.pad(y, w // 2, mode='edge')                 # edge-pad → no boundary droop
    return np.convolve(yp, np.ones(w) / w, mode='valid')[:len(y)]


def thr_ep(eps, y, thr):
    hit = np.where(y >= thr)[0]
    return int(eps[hit[0]]) if len(hit) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vqc', default='results/result_90')
    ap.add_argument('--dnn', default='results/result_95')
    ap.add_argument('--cap-vqc', type=int, default=2000)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--out', default='analysis/data/sample_eff_c2.png')
    a = ap.parse_args()

    ve, vq, vr = parse_log(f'{a.vqc}/training_log.txt', a.K, cap=a.cap_vqc)
    de, dq, dr = parse_log(f'{a.dnn}/training_log.txt', a.K)
    print(f"VQC {a.vqc}: {len(ve)} eps (trimmed @{a.cap_vqc})  ·  DNN {a.dnn}: {len(de)} eps")

    # ── (D) threshold-reaching + convergence summary ─────────────────────────
    print("\n(D) THRESHOLD-REACHING (smoothed, w=25):")
    vqs, dqs = smooth(vq), smooth(dq)
    vrs, drs = smooth(vr), smooth(dr)
    for name, thr in [("QoS>=0.90", 0.90), ("QoS>=0.95", 0.95)]:
        print(f"  {name:12s}  VQC ep {thr_ep(ve, vqs, thr)}   DNN ep {thr_ep(de, dqs, thr)}")
    print(f"  final QoS   (last 200 ep): VQC {100*vq[-200:].mean():.1f}%  DNN {100*dq[-200:].mean():.1f}%")
    print(f"  final R_tot (last 200 ep): VQC {vr[-200:].mean():.4f}   DNN {dr[-200:].mean():.4f}")
    print(f"  QoS variability (std, full run): VQC {100*vq.std():.2f}pp  DNN {100*dq.std():.2f}pp")

    # ── (A) learning-curve figure ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax, (ve_, vy, de_, dy, lab) in [
        (ax1, (ve, vq, de, dq, "QoS fraction")),
        (ax2, (ve, vr, de, dr, r"$R_{\mathrm{tot}}$ (bps/Hz)")),
    ]:
        ax.plot(ve_, vy, color='C0', alpha=0.18, lw=0.6)
        ax.plot(de_, dy, color='C1', alpha=0.18, lw=0.6)
        ax.plot(ve_, smooth(vy), color='C0', lw=2.0, label='HQC-HAC (VQC)')
        ax.plot(de_, smooth(dy), color='C1', lw=2.0, label='DNN')
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    ax1.legend(loc='lower right'); ax1.set_title(r'Sample efficiency at $R_{\mathrm{LoS}}=0.2$ (Case 2)')
    ax2.set_xlabel('PPO episode')
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"\nsaved figure -> {a.out}")


if __name__ == '__main__':
    main()
