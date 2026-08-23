#!/bin/bash
# Short-lived parent that hands the no-entangle waiter its own session.
# NO arguments: PowerShell Start-Process -ArgumentList splits on spaces.
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
setsid bash analysis/data/_waiter_noentangle.sh </dev/null >/dev/null 2>&1 &
sleep 8
