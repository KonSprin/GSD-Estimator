# %%
import pyopencl as cl
import numpy as np
from timeit import default_timer as timer
from gsdest import *
import logging

logging.basicConfig(filename='gsd.log', filemode='w', 
                    level=logging.DEBUG,
                    format='%(levelname)s: %(module)s %(asctime)s %(message)s')

# %% Create a context
timer_start = timer()

queue, cntxt = create_context()

timer_context = timer()
print("Created context: " + str(timer_context - timer_start))

# %% Load the prob grid
grid = load_grid()

timer_grid = timer()
print("Loaded Grid: " + str(timer_grid - timer_context))

# %% Load the scores
scores = load_scores("scores.npy")

timer_scores = timer()
print("Loaded Scores: " + str(timer_scores - timer_grid))

# %% create the buffers to hold the values of the input
grid_length = int(grid.shape[0]/10)
scores_length = scores.shape[0]
out = np.empty((grid_length,scores_length), dtype=np.float32)

grid_buf, scores_buf, out_buf = create_buffers(cntxt, grid, scores, out)

timer_buffers = timer()
print("Created Buffers: " + str(timer_buffers - timer_scores))

# %% Kernel
estimator_prog = estimator_program(cntxt, scores_length)

timer_kernel = timer()
print ("Builded kernel: " + str(timer_kernel - timer_buffers))

# %%
estimator_prog(queue, grid.shape, None, grid_buf, scores_buf, out_buf)
cl.enqueue_copy(queue, out, out_buf)

timer_calc = timer()
print ("calculaiton time: " + str(timer_calc - timer_kernel))

# %%
max_likelihood_idx = find_max_likelihoods(out, scores_length)

# %%
save_results(max_likelihood_idx, grid, out)

timer_end = timer()
print ("Finding max: " + str(timer_end - timer_calc))

summary_time = timer_end - timer_start
print ("Summary: " + str(summary_time))
print ("Per sample: " + str(summary_time/scores_length) + " for " + str(scores_length) + " samples")