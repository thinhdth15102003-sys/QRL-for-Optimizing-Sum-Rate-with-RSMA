"""
replay.py
---------
Simple circular experience replay buffer.

Each transition stored as a dict so that any-shape arrays can be replayed:
  's_t', 's_t_next'   — full actor state vectors
  'phi'                — IRS assignment (K,) int
  's_phase', 'phase_idx'           — PhaseMLP input + sampled action
  's_power', 'fracs_power', 'active_mask'   — PowerMLP input + sample + mask
  's_ck', 'alpha_k'                — CkMLP input + within-group probs
  'reward', 'is_terminal'

At sample time the caller recomputes V(s_t), V(s_{t+1}), advantage and the
policy-gradient with the CURRENT critic and actor (approximate-on-policy:
the policy ratio is treated as 1 — bias is small when buffer is small and
the policy changes slowly).
"""

import numpy as np
from typing import Optional


class ReplayBuffer:
    def __init__(self, capacity: int,
                 rng: Optional[np.random.Generator] = None):
        self.capacity   = int(capacity)
        self.buffer:    list = []
        self.pos:       int  = 0
        self.rng              = rng or np.random.default_rng()

    def push(self, transition: dict) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        n   = min(len(self.buffer), int(batch_size))
        idx = self.rng.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)
