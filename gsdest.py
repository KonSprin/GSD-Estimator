import pyopencl as cl
import numpy as np
import pandas
import sys
import csv

def create_context():
  platforms = cl.get_platforms()
  cntxt = cl.Context(
          dev_type=cl.device_type.ALL,
          properties=[(cl.context_properties.PLATFORM, platforms[0])])

  queue = cl.CommandQueue(cntxt)

  return (queue, cntxt)

def load_grid(filename="gsd_prob_grid.npy"):
  try:
    grid = np.load(filename)
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
  return grid

def load_scores(filename="scores.npy"):
  try: 
    scores = np.load(filename)
    print("Succesfully loaded scores from file")
  except IOError:
    try:
      print("no .npy file, loading from csv")

      scores = np.genfromtxt("scores.csv", delimiter=',', dtype=int)
      zeros = np.zeros((scores.shape[0],3), dtype=int)
      scores = np.concatenate((scores,zeros), axis=1)
      np.save("scores.npy", scores)

      print("Succesfully loaded scores from csv")
    except IOError:
      print("no csv file. Please provide one")
      sys.exit(1)
  return scores

def create_buffers(cntxt, grid, scores, out):
  grid_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | 
  cl.mem_flags.COPY_HOST_PTR,hostbuf=grid)

  scores_buf = cl.Buffer(cntxt,  cl.mem_flags.READ_ONLY | 
  cl.mem_flags.COPY_HOST_PTR,hostbuf=scores)

  out_buf = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, out.nbytes)

  return (grid_buf, scores_buf, out_buf)

def estimator_program(cntxt, scores_length):
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
  n_samples = int(scores_length)
  bld = cl.Program(cntxt, code).build(options=['-D', 'n_samples='+str(n_samples)])
  estiamtor = bld.estimate

  return estiamtor

def find_max_likelihoods(out, scores_length):
  max_likelihood_idx = [0] * scores_length
  for i in range(scores_length):
    max_likelihood_idx[i] = out[:,i].argmax()
  return max_likelihood_idx

def save_results(max_likelihood_idx, grid, out):
  with open('results.csv', 'w', newline='') as results_file:
    writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['idx', 'psi', 'rho', 'log_likelihood'])
    for (i, idx) in enumerate(max_likelihood_idx):
      writer.writerow([i, grid[idx*10], grid[idx*10+1], out[idx,i]])
