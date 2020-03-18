#!/usr/bin/env bash

source activate ml
dir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $dir/experiments


for task in 0 1 2 3 4 5 6 7 8 9 10 11
do
    run=0
    chain=$((task / 3))
    true=$((task % 3))

    echo ""
    echo ""
    echo ""
    echo "Starting new job: 48d with true = ${true}, chain = ${chain}, run = ${run}"
    echo ""
    python -u evaluate.py --modelname april --dataset lhc --algorithm flow --modellatentdim 14 --observedsamples 50 --splinebins 10 -i $run --skiplikelihood --burnin 50 --mcmcsamples 750 --trueparam $true --chain $chain --dir $dir

    echo ""
    echo ""
    echo ""
    echo "Starting new job: 2d with true = ${true}, chain = ${chain}, run = ${run}"
    echo ""
    python -u evaluate.py --modelname april --dataset lhc2d --algorithm flow --modellatentdim 2 --observedsamples 50 --splinebins 10 -i $run --skiplikelihood --burnin 50 --mcmcsamples 750 --trueparam $true --chain $chain --dir $dir
done
