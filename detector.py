import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal 
import time 
import threading
import numpy as np
import math
import scikits.audiolab

def SampleFrequencies(data, window_mids, window_size, freq_output, amp_output, output_start):
  for i, mid in enumerate(window_mids):
    freq_output[output_start + i][:] = abs((np.fft.rfft(data[mid - (window_size/2) : mid + (window_size/2) + 1])/window_size))
    amp_output[output_start + i] = sum(abs(data[mid - (window_size/2) : mid + (window_size/2) + 1])/window_size)
    #amp_output[output_start + i] = freq_output[output_start + i][0]

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

def LogInterpolate(x1, x2, t):
  return (x2**t)*(x1**(1-t))

def PlotSignal(freqs, times=None):
  if times is not None:
    plt.plot(times, freqs)
  else:
    plt.plot(freqs)
  plt.show()

def GetDrop(file, play_file=False):
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
  target_window_duration = .05
  window_size = int(rate * target_window_duration)
  window_duration = float(window_size) / rate 
  if window_size % 2 is 0:
    window_size = window_size + 1
  n_windows = int((len(data) / window_size) * 1.5) # Get some sampling overlap
  sample_mids, freq_out, amp_out = SampleSong(data, n_windows, window_size)

  # Now, do something with the frequencies.
  n_bands = len(freq_out[0])
  n_subbands = 32
  base = 10
  subbands = np.logspace(1, math.log(n_bands + base, base), num=(n_subbands + 1), base=base) - base
  band_data = np.empty((n_windows, n_subbands))
  for i in xrange(n_windows):
    for j in xrange(n_subbands):
      band_data[i, j] = np.sum(freq_out[i, int(subbands[j]):int(subbands[j+1])])

  # So now we have roughly banded data. Normalize
  band_means = np.mean(band_data, axis=0)
  band_data = band_data - band_means
  band_std = np.std(band_data, axis=0) + 1e-10
  band_data = band_data / band_std
  
  # Lowpass filter on the scale of builds
  build_duration = 15.0
  macro_filter_size = int(build_duration/window_duration)
  if macro_filter_size % 2 is 0:
    macro_filter_size = macro_filter_size + 1
  macro_filter = scipy.signal.boxcar(macro_filter_size)
  macro_band_data = np.empty(band_data.shape)
  for band in xrange(n_subbands):
    macro_band_data[:, band] = scipy.signal.convolve(band_data[:, band], macro_filter, mode='same')

  # Lowpass filter on the scale of drops 
  drop_duration = .75
  micro_filter_size = int(drop_duration/window_duration)
  if micro_filter_size % 2 is 0:
    micro_filter_size = micro_filter_size + 1
  micro_filter = scipy.signal.boxcar(micro_filter_size)
  micro_band_data = np.empty(band_data.shape)
  for band in xrange(n_subbands):
    micro_band_data[:, band] = scipy.signal.convolve(band_data[:, band], micro_filter, mode='same')


  # Weight bass highly
  subband_weights = np.logspace(2.0, 1.0, n_subbands)

  correlation = np.empty(n_windows)
  start = int(max(micro_filter_size/2, macro_filter_size/2))
  # Find points with low correlation between the two filters
  for i in xrange(start, n_windows - start):
    # Correlations, ie. "how similar are these two signals", i.e. "How similar is the macro frequency distribution to the micro?"
    pre_corr = micro_band_data[i + micro_filter_size/2, :] * macro_band_data[i - macro_filter_size/2, :]
    post_corr = micro_band_data[i - micro_filter_size/2, :] * macro_band_data[i + macro_filter_size/2, :]

    # Also get the sign of the change, so we can detect increases vs. decreases
    pre_delta = np.sign(macro_band_data[i - macro_filter_size/2, :] - micro_band_data[i + micro_filter_size/2, :])
    post_delta = np.sign(macro_band_data[i + macro_filter_size/2, :] - micro_band_data[i - micro_filter_size/2, :])

    # Apply weights, and sum
    correlation[i] = sum(pre_corr * pre_delta * subband_weights) + sum(post_corr * post_delta * subband_weights)


  #correlation[i] = sum(pre_corr * pre_delta * subband_weights) + sum(post_corr * post_delta * subband_weights)
  #correlation[0:-1] =  correlation[1:] - correlation[0:-1]
  filter = [-1.0] * micro_filter_size
  filter.append(0)
  filter.extend([1.0] * micro_filter_size)

  #correlation = scipy.signal.correlate(correlation, np.linspace(-1.0, 1.0, micro_filter_size), mode='same')
  correlation = scipy.signal.correlate(correlation, filter, mode='same')


  #start = int(((0 - micro_filter_size/2) + macro_filter_size/2))
  #for i in xrange(start, n_windows - start):
  #  pre_corr = micro_band_data[i + micro_filter_size/2, :] * macro_band_data[i - macro_filter_size/2, :]
  #  post_corr = micro_band_data[i - micro_filter_size/2, :] * macro_band_data[i + macro_filter_size/2, :]
  #  pre_delta = np.sign(macro_band_data[i - macro_filter_size/2, :] - micro_band_data[i + micro_filter_size/2, :])
  #  post_delta = np.sign(macro_band_data[i + macro_filter_size/2, :] - micro_band_data[i - micro_filter_size/2, :])
  #  correlation[i] = sum(pre_corr * pre_delta * subband_weights) + sum(post_corr * post_delta * subband_weights)

  result = np.argmax(correlation[start:-start]) + start
  print sample_mids[result] / float(data.shape[0]) * song_duration

  before_drop = (data[:sample_mids[result]]/float(2**17)).T
  after_drop = (data[sample_mids[result]:]/float(2**17)).T
  if play_file:
    scikits.audiolab.play(before_drop, fs=rate)


  silent_result = np.argmin(amp_out[start:-start]) + start
  print song_duration * result / n_windows
  print song_duration * silent_result / n_windows
  times = (np.arange(n_windows) / float(n_windows)) * song_duration
  PlotSignal(correlation, times)
  if play_file:
    scikits.audiolab.play(after_drop, fs=rate)

  #freqs = np.repeat([np.log(np.arange(window_size/2 + 1) / window_duration + 1e-10)], n_windows, axis=0)
  #avg_freq = np.average(freqs, weights=freq_out, axis=1)

  #freqs = np.arange(window_size/2 + 1) / window_duration

  #plt.imshow(np.rot90(freq_out), extent=(0.0, song_duration, freqs[0][0], freqs[0][-1]), aspect=(song_duration/freqs[0][-1]), cmap=plt.get_cmap('hot'), interpolation='nearest')
  #plt.plot(times, amp_out)
  #plt.plot(times, avg_freq)
  #plt.show()

  #for window in out:
  #  print np.argmax(window) / window_duration, max(window), min(window)
  #print window_duration

GetDrop('shortbloodred.wav')
GetDrop('beam_mako.wav')
#GetDrop('overtime.wav')
#GetDrop('bloodred.wav')
#GetDrop('louder.wav')
#Heatmap('440hz.wav')
