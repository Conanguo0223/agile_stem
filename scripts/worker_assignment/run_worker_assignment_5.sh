#!/bin/bash

INFO="[('A0', 'n_workers'), ('A1', 'n_workers'), ('A2', 'n_workers'), ('A3', 'n_workers'), ('A4', 'n_workers')]"
ENV="worker_assignment"
TOTAL_STEPS=500000
N=5

for SEED in 42 2024 174
do
    ../run_ppo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    ../run_rppo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    ../run_a2c.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    ../run_trpo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
done
