# %%

import pyopencl as cl
import numpy as np
from timeit import default_timer as timer
import pandas
import sys
import matplotlib.pyplot as plt
import csv

# %% Create a context
timer_start = timer()
platforms = cl.get_platforms()
cntxt = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])

# now create a command queue in the context
queue = cl.CommandQueue(cntxt)

timer_context = timer()
print("Created context: " + str(timer_context - timer_start))

# %% Load the prob grid
try:
  grid = np.load("gsd_prob_grid.npy")
  print("Succesfully loaded grid from file")
except IOError:
  print("no numpy file, trying pandas")
  try:
    dataframe = pandas.read_pickle("gsd_prob_grid.pkl")
          
    grid_py = []

    for (psi, ro), row in dataframe.iterrows():
        grid_py.append(psi)
        grid_py.append(ro)
        for idx in range(1, 6):
          grid_py.append(row[idx])
        for z in range(3):
          grid_py.append(0)

    grid = np.array(grid_py, dtype=np.float32)
    np.save('gsd_prob_grid.npy', grid)
  except IOError:
      print("Pandas file does not exists")
      sys.exit(1)

timer_grid = timer()
print("Loaded Grid: " + str(timer_grid - timer_context))

# grid2d = np.resize(grid, (int(grid.shape[0]/10), 10))
# plt.plot(grid2d[2,:])
# plt.show()
# sys.exit(0)

# %% Load the scores
try: 
  scores = np.load("scores.npy")
  print("Succesfully loaded scores from file")
except IOError:
  print("no scores file, generating sample file")

  scores_py = [[1618, 1486, 266, 50, 76, 0, 0, 0],
            [75365, 164164, 55766, 4705, 6490, 0, 0, 0],
            [5097, 7645, 30583, 198794, 265059, 0, 0, 0],
            [265059, 198794, 30583, 7645, 5097, 0, 0, 0]]
  
  tmp = []

  for a in range(200):
    for score in scores_py:
      tmp.append(score)
  
  scores = np.array(tmp, dtype=int)

  np.save("scores.npy", scores)

timer_scores = timer()
print("Loaded Scores: " + str(timer_scores - timer_grid))

grid_length = int(grid.shape[0]/10)
scores_length = scores.shape[0]
out = np.empty(scores_length*grid_length, dtype=np.float32)

# %% create the buffers to hold the values of the input
grid_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | 
cl.mem_flags.COPY_HOST_PTR,hostbuf=grid)

scores_buf = cl.Buffer(cntxt,  cl.mem_flags.READ_ONLY | 
cl.mem_flags.COPY_HOST_PTR,hostbuf=scores)

out_buf = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, out.nbytes)

timer_buffers = timer()
print("Created Buffers: " + str(timer_buffers - timer_scores))

# Kernel Program
code = """
typedef struct __attribute__ ((packed)) { \
  float psi; \
  float ro; \
  float grades[8]; \
} grid_node; \
 \
__kernel void \
estimate( \
         __global grid_node *node, \
         __global int *samples_ptr, \
         __global float *out) \
{ \
  size_t id = get_global_id(0); \
  float8 probs = log(vload8(0, node[id].grades)); \
  for (int i = 0; i < n_samples; ++i) { \
    int8 samples = vload8(i, samples_ptr); \
    float8 res = probs * convert_float8(samples); \
    float sum = res.s0 + res.s1 + res.s2 + res.s3 + res.s4; \
    out[n_samples*id + i] = sum; \
  } \
} \
"""

# %% build the Kernel
n_samples = int(scores_length)
bld = cl.Program(cntxt, code).build(options=['-D', 'n_samples='+str(n_samples)])
# Kernel is now launched
launch = bld.estimate

timer_kernel = timer()
print ("Builded kernel: " + str(timer_kernel - timer_buffers))

# %%
launch(queue, grid.shape, None, grid_buf, scores_buf, out_buf)

cl.enqueue_copy(queue, out, out_buf)

timer_calc = timer()
print ("calculaiton time: " + str(timer_calc - timer_kernel))

# %%
max_likelihood = [float('-inf')] * scores_length
max_likelihood_idx = [0] * scores_length
for i in range(grid_length):
  for n in range(scores_length):
    if (out[i * scores_length + n] > max_likelihood[n]):
      max_likelihood[n] = out[i * scores_length + n]
      max_likelihood_idx[n] = i

# %%

with open('results.csv', 'w', newline='') as results_file:
  writer = csv.writer(results_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
  writer.writerow(['idx', 'psi', 'rho', 'log_likelihood'])
  for (i, idx) in enumerate(max_likelihood_idx):
    writer.writerow([i, grid[idx*10], grid[idx*10+1], max_likelihood[i]])

# print the output
# print ("Grid :", grid)
# print ("Output :", out)
# print (grid.shape)
# print (out.shape)

timer_end = timer()
print ("Finding max: " + str(timer_end - timer_calc))

summary_time = timer_end - timer_start
print ("Summary: " + str(summary_time))
print ("Per sample: " + str(summary_time/scores_length) + " for " + str(scores_length) + " samples")