#!/bin/bash
# Stage 4: all figures and the baseline table -> figs/ (conda env rescp).
set -e; cd "$(dirname "$0")/.."
python figures/make_figures.py --results results --results-sim results_sim/closed_2011 --data data --out figs
