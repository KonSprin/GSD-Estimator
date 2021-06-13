import pyopencl as cl
import numpy as np
import logging
import pandas
import sys
import csv

def create_context():
  logging.debug("starting to create context")
  platforms = cl.get_platforms()
  cntxt = cl.Context(
          dev_type=cl.device_type.ALL,
          properties=[(cl.context_properties.PLATFORM, platforms[0])])
  logging.debug("Created context")

  queue = cl.CommandQueue(cntxt)
  logging.info("Sucessfully created queue and context")

  return (queue, cntxt)

def load_grid(filename="gsd_prob_grid.npy"):
  try:
    grid = np.load(filename)
    logging.info("Succesfully loaded grid from file")
  except:
    logging.warning("no numpy file, trying pandas")
    try:
      dataframe = pandas.read_pickle(filename)
      logging.debug("Loaded grid from pickle file, converting to numpy format")

      grid_py = []

      for (psi, ro), row in dataframe.iterrows():
          grid_py.append(psi)
          grid_py.append(ro)
          for val in row:
            grid_py.append(val)
          for z in range(3):
            grid_py.append(0)

      grid = np.array(grid_py, dtype=np.float32)
      logging.info("Succesfuly converted grid to numpy format, saving to file")
      np.save('gsd_prob_grid.npy', grid)
    except IOError:
        logging.error("Pandas file does not exists")
        sys.exit(1)
  return grid

def load_scores(filename="scores.npy"):
  try: 
    scores = np.load(filename)
    logging.info("Succesfully loaded %i scores from file", scores.shape[0])
  except:
    logging.warning("no .npy file, trying from csv")
    try:
      scores = np.genfromtxt(filename, delimiter=' ', dtype=int)
      logging.info("Succesfully loaded scores from csv")

      zeros = np.zeros((scores.shape[0],3), dtype=int)
      scores = np.concatenate((scores,zeros), axis=1)
      np.save("scores.npy", scores)

      logging.info("Saved scores to npy file for future")
    except np.AxisError:
      logging.warning(" \" \" is not the delimiter. Trying \",\"" )

      scores = np.genfromtxt(filename, delimiter=',', dtype=int)
      logging.info("Succesfully loaded scores from csv")

      zeros = np.zeros((scores.shape[0],3), dtype=int)
      scores = np.concatenate((scores,zeros), axis=1)
      np.save("scores.npy", scores)

      logging.info("Saved scores to npy file for future")
    except IOError:
      logging.error("no csv file. Please provide one")
      sys.exit(1)
  return scores

def create_buffers(cntxt, grid, scores, out):
  logging.info("Starting to create buffers")

  grid_buf = cl.Buffer(cntxt, cl.mem_flags.READ_ONLY | 
  cl.mem_flags.COPY_HOST_PTR,hostbuf=grid)
  logging.debug("Succesfully created grid buffer")

  scores_buf = cl.Buffer(cntxt,  cl.mem_flags.READ_ONLY | 
  cl.mem_flags.COPY_HOST_PTR,hostbuf=scores)
  logging.debug("Succesfully created scores buffer")

  out_buf = cl.Buffer(cntxt, cl.mem_flags.WRITE_ONLY, out.nbytes)
  logging.debug("Succesfully created output buffer")

  logging.info("All buffers created")
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

  logging.info("Succesfully created kernel")
  return estiamtor

def find_max_likelihoods(out, scores_length):
  logging.debug("Starting the finding max loop")

  max_likelihood_idx = [0] * scores_length
  for i in range(scores_length):
    max_likelihood_idx[i] = out[:,i].argmax()

  logging.info("Finished finding max likelihoods")
  return max_likelihood_idx

def save_results(max_likelihood_idx, grid, out):
  logging.debug("Saving teh results to the results.csv file")
  with open('results.csv', 'w', newline='') as results_file:
    writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['idx', 'psi', 'rho', 'log_likelihood'])
    for (i, idx) in enumerate(max_likelihood_idx):
      writer.writerow([i, grid[idx*10], grid[idx*10+1], out[idx,i]])
  logging.info("Saved results to file")


def start_from_file(scores_filename="scores.npy", grid_filename="gsd_prob_grid.npy"):
  queue, cntxt = create_context()
  grid = load_grid(grid_filename)
  scores = load_scores(scores_filename)

  grid_length = int(grid.shape[0]/10)
  logging.info("Grid length is: " + str(grid_length))

  scores_length = scores.shape[0]
  logging.info("There is " + str(scores_length) + " Scores")

  out = np.empty((grid_length,scores_length), dtype=np.float32)

  grid_buf, scores_buf, out_buf = create_buffers(cntxt, grid, scores, out)

  estimator_prog = estimator_program(cntxt, scores_length)

  logging.info("Starting calculations")
  estimator_prog(queue, grid.shape, None, grid_buf, scores_buf, out_buf)
  cl.enqueue_copy(queue, out, out_buf)
  logging.info("Finished calculations")

  max_likelihood_idx = find_max_likelihoods(out, scores_length)

  save_results(max_likelihood_idx, grid, out)

def start(scores, grid):
  queue, cntxt = create_context()

  grid_length = int(grid.shape[0]/10)
  logging.info("Grid length is: " + str(grid_length))

  scores_length = scores.shape[0]
  logging.info("There is " + str(scores_length) + " Scores")

  out = np.empty((grid_length,scores_length), dtype=np.float32)

  grid_buf, scores_buf, out_buf = create_buffers(cntxt, grid, scores, out)

  estimator_prog = estimator_program(cntxt, scores_length)

  logging.info("Starting calculations")
  estimator_prog(queue, grid.shape, None, grid_buf, scores_buf, out_buf)
  cl.enqueue_copy(queue, out, out_buf)
  logging.info("Finished calculations")

  max_likelihood_idx = find_max_likelihoods(out, scores_length)

  save_results(max_likelihood_idx, grid, out)