import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.io import wavfile
import time 
import threading
import numpy as np

def SampleFrequencies(data, window_mids, window_size, freq_output, amp_output, output_start):
  for i, mid in enumerate(window_mids):
    freq_output[output_start + i][:] = abs((np.fft.rfft(data[mid - (window_size/2) : mid + (window_size/2) + 1])/window_size))
    amp_output[output_start + i] = sum(abs(data[mid - (window_size/2) : mid + (window_size/2) + 1])/window_size)

def SampleSong(data, n_samples, window_size):
  assert window_size % 2 is 1, 'Window size must be odd'
  freq_output = np.empty((n_samples, window_size/2 + 1)) # Because this is a real DFT, the DFT output is of size n/2 + 1
  amp_output = np.empty(n_samples) # Because this is a real DFT, the DFT output is of size n/2 + 1
  sample_mids = np.linspace(window_size/2, len(data) - (window_size/2 + 1), n_samples, dtype=int) 
  SampleFrequencies(data, sample_mids, window_size, freq_output, amp_output, 0)
  return sample_mids, freq_output, amp_output


#def FindSilence(data, min_length):
#def FindPeakForward(data, min_length):
#def FindPeakBackward(data, min_length):
#def FindPeakFrequency(data, min_length):
#def FindPeakFrequency(data, min_length):
  

def Heatmap(file):
  # Read the file
  rate, data = wavfile.read(file)

  # Only bother with one channel for now
  n_samples = data.shape[0]
  if len(data.shape) > 1:
    n_channels = data.shape[1]
    data = data[:, 0]
  else:
    n_channels = 1

  # Figure out how we want to sample the samples
  song_duration = float(len(data)) / rate
  target_window_duration = .1
  window_size = int(rate * target_window_duration)
  if window_size % 2 is 0:
    window_size = window_size + 1
  n_windows = int((len(data) / window_size) * 1.5) # Get some sampling overlap
  sample_mids, freq_out, amp_out = SampleSong(data, n_windows, window_size)

  window_duration = float(window_size) / rate 
  freqs = np.arange(window_size/2) / window_duration
  times = sample_mids * (song_duration / n_samples)


  #plt.imshow(np.rot90(freq_out), extent=(0.0, song_duration, freqs[0], freqs[-1]), aspect=(song_duration/freqs[-1]), cmap=plt.get_cmap('hot'), interpolation='nearest')
  #plt.plot(times, amp_out)
  #plt.show()

  #for window in out:
  #  print np.argmax(window) / window_duration, max(window), min(window)
  #print window_duration

#Heatmap('bloodred.wav')
Heatmap('louder.wav')
#Heatmap('440hz.wav')
