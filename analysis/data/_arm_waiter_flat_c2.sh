#!/bin/bash
# Short-lived parent whose only job is to hand the waiter its own session.
# Must be invoked with NO arguments: PowerShell's Start-Process -ArgumentList
# splits on spaces, so any space-bearing argument would be torn apart.
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
setsid bash analysis/data/_waiter_flat_c2_nshift.sh </dev/null >/dev/null 2>&1 &
sleep 8
