#!/bin/bash
cd /home/golobs/SpectralPrivacy
PY=~/miniconda3/envs/recon_/bin/python
for ds in adult cdc_diabetes california; do
  CUDA_VISIBLE_DEVICES=1 nice -n 5 $PY tabddpm_experiment.py features --dataset $ds && \
  nice -n 5 $PY tabddpm_experiment.py analyze --dataset $ds --k 10
  echo "DONE $ds rc=$?"
done
