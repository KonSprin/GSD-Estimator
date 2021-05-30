import pyopencl as cl
import numpy as np
from gsdest import *

def start(scores_filename="scores.npy", grid_filename="gsd_prob_grid.npy"):
  queue, cntxt = create_context()
  grid = load_grid(grid_filename)
  scores = load_scores(scores_filename)

  grid_length = int(grid.shape[0]/10)
  scores_length = scores.shape[0]
  out = np.empty((grid_length,scores_length), dtype=np.float32)

  grid_buf, scores_buf, out_buf = create_buffers(cntxt, grid, scores, out)

  estimator_prog = estimator_program(cntxt, scores_length)

  estimator_prog(queue, grid.shape, None, grid_buf, scores_buf, out_buf)
  cl.enqueue_copy(queue, out, out_buf)

  max_likelihood_idx = find_max_likelihoods(out, scores_length)

  save_results(max_likelihood_idx, grid, out)