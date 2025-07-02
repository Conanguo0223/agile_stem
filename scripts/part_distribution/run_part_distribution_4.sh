#!/bin/bash

INFO="[('SwitchD', 'index_buffer_out')]"
ENV="part_distribution"
TOTAL_STEPS=10000000
N=4

for SEED in 42 2024 174
do
    ../run_ppo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    ../run_rppo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    ../run_a2c.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    ../run_trpo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
done
