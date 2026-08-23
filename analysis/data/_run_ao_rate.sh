#!/bin/bash
cd '/mnt/c/Project/IRS-assisted RSMA Quantum-RL'
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
echo "########## CASE-1 (K5 M1) @R_LoS0.5 ##########"
$PY analysis/probe_ao_rate.py --run results/result_117 --R-LoS 0.5 --episodes 50 --steps 200 --seed 42
echo "########## CASE-2 (K10 M2) @R_LoS0.5 ##########"
$PY analysis/probe_ao_rate.py --run results/result_101 --R-LoS 0.5 --episodes 50 --steps 200 --seed 42
