#!/bin/bash

ENV="complex_line"
INFO="[]"
TOTAL_STEPS=100000000
N=3

for SEED in 42 2024 174
do
    ./run_rppo.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    ./run_rppo_curriculum.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
    ./run_ppo_curriculum.sh $SEED $ENV $N "$INFO" $TOTAL_STEPS
done
