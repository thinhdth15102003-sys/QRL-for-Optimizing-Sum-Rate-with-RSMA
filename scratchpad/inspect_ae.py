"""Break down VQC-actor params into Encoder / Decoder / VQC-core (+ any post/readout)
for the param-eff table. Categorize actor_params.npz keys."""
import numpy as np

RUNS = [("C1 (5,1) r117", "results/result_117/checkpoints/ep_01900/agents/actor_params.npz"),
        ("C2 (10,2) r110", "results/result_110/checkpoints/ep_01600/agents/actor_params.npz")]
for tag, path in RUNS:
    d = np.load(path)
    enc = dec = other = 0
    print(f"=== {tag} ===")
    for k, v in sorted(d.items()):
        kl = k.lower()
        cat = 'ENC' if ('enc' in kl) else ('DEC' if ('dec' in kl) else 'CORE')
        if cat == 'ENC': enc += v.size
        elif cat == 'DEC': dec += v.size
        else: other += v.size
        print(f"  [{cat:4}] {k:22} {v.size}")
    tot = enc + dec + other
    print(f"  → Encoder={enc}  Decoder={dec}  VQC-core/other={other}  | inference(core+enc)={other+enc}  train(all)={tot}\n")
