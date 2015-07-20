
class SampleThread(threading.Thread):
  def __init__(self, data, offset, n_windows, window_size, rate, output):
    threading.Thread.__init__(self)
    self.data = data
    self.offset = offset 
    self.n_windows = n_windows
    self.window_size = window_size
    self.rate = rate
    self.output = output

  def run(self):
    SampleFrequencies(self.data, self.offset, self.n_windows, self.window_size, self.rate, self.output)
    


def SampleSongSMP(data, window_size, rate):
  n_threads = 4
  n_windows = len(data) / window_size
  print n_windows
  windows_per_thread = n_windows / n_threads
  left_overs = n_windows % n_threads
  sampling_threads = [] 
  offset = 0
  output = np.empty((n_windows, window_size/2))
  for i in xrange(n_threads):
    if i < left_overs:
      thread_n_windows = windows_per_thread + 1
    else:
      thread_n_windows = windows_per_thread
    if i == n_threads - 1:
      SampleFrequencies(data, offset, offset + thread_n_windows, window_size, rate, output)
    else:
      sampling_threads.append(SampleThread(data, offset, offset + thread_n_windows, window_size, rate, output))
      sampling_threads[-1].start()
    offset = offset + thread_n_windows
  for thread in sampling_threads:
    thread.join()
  return output
