# GSD-Estimator

Estimator for [Generalized Score Distibution's](https://arxiv.org/abs/1909.04369) Distibution's parameters using OpenCL for accelerating calculation on GPI.

## Requirements

> pip install -r requirements.txt

## Usage

* Script takes `scores.csv` file and `gsd_prob_grid` in pandas or numpy format as input
* output file will be a csv file with results decribed by values
  * idx - index of sample
  * psi - estimated mean (rating)
  * rho - estimated variance
  * log_likelihood - logarithm of likelihood value

## TODO

1. Optimize finding max likelihood loop to be less bad
2. Manage memory to not allocate too much in VRAM 
3. Manage input values not to use default files
4. Refactor main to functions