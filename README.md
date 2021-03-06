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
* While `main.py` is implementation with timers and debug prints, the same can be achieved with one simple function `gsdest.start()` inside another code that takes only samples and grid files as input and handles everything else:
  ```python
  import gsdest

  gsdest.start()
  ```

## TODO

1. ~~Optimize finding max likelihood loop to be less bad~~ <span style="color:green">DONE</span>
2. Manage memory to not allocate too much in VRAM 
3. ~~Manage input values not to use default files~~ <span style="color:green">DONE</span>
4. ~~Refactor main to functions~~ <span style="color:green">DONE</span>
5. ~~Make normal logger instead of simple prints~~ <span style="color:green">DONE</span>