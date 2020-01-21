#!/usr/bin/env bash

source activate ml
basedir=/Users/johannbrehmer/work/projects/manifold_flow/manifold-flow
cd $basedir/experiments

python -u evaluate.py --dataset spherical_gaussian --algorithm flow --epsilon 0.01 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm flow --epsilon 0.001 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm flow --epsilon 0.1 --dir $basedir

python -u evaluate.py --dataset spherical_gaussian --algorithm gamf --epsilon 0.01 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm gamf --epsilon 0.001 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm gamf --epsilon 0.1 --dir $basedir

python -u evaluate.py --dataset spherical_gaussian --algorithm mf --epsilon 0.01 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm mf --epsilon 0.001 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm mf --epsilon 0.1 --dir $basedir

python -u evaluate.py --dataset spherical_gaussian --algorithm pie --epsilon 0.01 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm pie --epsilon 0.001 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm pie --epsilon 0.1 --dir $basedir

python -u evaluate.py --dataset spherical_gaussian --algorithm mf --specified --dropout 0 --epsilon 0.01 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm mf --specified --dropout 0 --epsilon 0.001 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm mf --specified --dropout 0 --epsilon 0.1 --dir $basedir

python -u evaluate.py --dataset spherical_gaussian --algorithm gamf --specified --dropout 0 --epsilon 0.01 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm gamf --specified --dropout 0 --epsilon 0.001 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm gamf --specified --dropout 0 --epsilon 0.1 --dir $basedir

python -u evaluate.py --dataset spherical_gaussian --algorithm pie --specified --dropout 0 --epsilon 0.01 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm pie --specified --dropout 0 --epsilon 0.001 --dir $basedir
python -u evaluate.py --dataset spherical_gaussian --algorithm pie --specified --dropout 0 --epsilon 0.1 --dir $basedir
