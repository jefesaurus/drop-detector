import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal 
import time 
import threading
import numpy as np
import math
import pyaudio

def SampleFrequencies(data, window_mids, window_size, freq_output, amp_output, output_start):
  for i, mid in enumerate(window_mids):
    freq_output[output_start + i][:] += abs((np.fft.rfft(data[mid - (window_size/2) : mid + (window_size/2) + 1])/window_size))
    amp_output[output_start + i] += np.sum(abs(data[mid - (window_size/2) : mid + (window_size/2) + 1])/window_size)
    #amp_output[output_start + i] = freq_output[output_start + i][0]

def SampleSong(data, n_samples, window_size):
  assert window_size % 2 is 1, 'Window size must be odd'

  freq_output = np.zeros((n_samples, window_size/2 + 1)) # Because this is a real DFT, the DFT output is of size n/2 + 1
  amp_output = np.zeros(n_samples) # Because this is a real DFT, the DFT output is of size n/2 + 1
  sample_mids = np.linspace(window_size/2, len(data) - (window_size/2 + 1), n_samples, dtype=int) 

  # Sum up frequency distributions and amplitude over channels
  for channel in range(data.shape[1]):
    SampleFrequencies(data[:,channel], sample_mids, window_size, freq_output, amp_output, 0)
  return sample_mids, freq_output, amp_output

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

  # Get mono and stereo to behave
  if len(data.shape) == 1:
    data = np.array([data]).T
  else:
    assert len(data.shape) == 2, 'Music file dimensionality is weird'
    data = data

  n_samples = data.shape[0]
  n_channels = data.shape[1]

  # Figure out how we want to sample the samples
  song_duration = n_samples / float(rate)
  TARGET_WINDOW_DURATION = .05
  window_size = int(rate * TARGET_WINDOW_DURATION)
  window_duration = float(window_size) / rate 
  if window_size % 2 is 0:
    window_size = window_size + 1
  n_windows = int((n_samples / window_size) * 1.5) # Get some sampling overlap
  sample_mids, freq_out, amp_out = SampleSong(data, n_windows, window_size)

  # Now, do something with the frequencies.
  n_bands = len(freq_out[0])
  N_SUBBANDS = 32
  base = 10
  subbands = np.logspace(1, math.log(n_bands + base, base), num=(N_SUBBANDS + 1), base=base) - base
  band_data = np.empty((n_windows, N_SUBBANDS))
  for i in xrange(n_windows):
    for j in xrange(N_SUBBANDS):
      band_data[i, j] = np.sum(freq_out[i, int(subbands[j]):int(subbands[j+1])])

  # So now we have roughly banded data. Normalize
  band_means = np.mean(band_data, axis=0)
  band_data = band_data - band_means
  band_std = np.std(band_data, axis=0) + 1e-10
  band_data = band_data / band_std
  
  # Lowpass filter on the scale of builds
  BUILD_DURATION = 15.0
  macro_filter_size = int(BUILD_DURATION/window_duration)
  if macro_filter_size % 2 is 0:
    macro_filter_size = macro_filter_size + 1
  macro_filter = scipy.signal.boxcar(macro_filter_size)
  macro_band_data = np.empty(band_data.shape)
  for band in xrange(N_SUBBANDS):
    macro_band_data[:, band] = scipy.signal.convolve(band_data[:, band], macro_filter, mode='same')

  # Lowpass filter on the scale of drops 
  DROP_DURATION = 1.0
  micro_filter_size = int(DROP_DURATION/window_duration)
  if micro_filter_size % 2 is 0:
    micro_filter_size = micro_filter_size + 1
  micro_filter = scipy.signal.boxcar(micro_filter_size)
  micro_band_data = np.empty(band_data.shape)
  for band in xrange(N_SUBBANDS):
    micro_band_data[:, band] = scipy.signal.convolve(band_data[:, band], micro_filter, mode='same')

  # Weight bass highly
  BASS_LOG_FACTOR = 2.0
  subband_weights = np.logspace(BASS_LOG_FACTOR, 1.0, N_SUBBANDS)

  times = (np.arange(n_windows) / float(n_windows)) * song_duration
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
    correlation[i] = np.sum(pre_corr * pre_delta * subband_weights) + np.sum(post_corr * post_delta * subband_weights)


  # Goofy "derivative finding" filter, long term highpass..? idk
  delta_filter = [-1.0] * micro_filter_size
  delta_filter.append(0)
  delta_filter.extend([1.0] * micro_filter_size)

  correlation = scipy.signal.correlate(correlation, delta_filter, mode='same')
  PlotSignal(correlation, times)

  # Correlate the weighted frequency changes with volume increases:
  amplitude_changes = scipy.signal.correlate(scipy.signal.convolve(amp_out, micro_filter, mode='same'), delta_filter, mode='same')
  PlotSignal(amplitude_changes, times)
  correlation = correlation * amplitude_changes * np.sign(amplitude_changes)

  # Correlate with a drop-likeliness ditribution over time:
  # Hypothesize that drops are most likely to occur in the middle 2/3 of the song
  RAMP_SIZE_DIV = float(6.0)
  RAMP_STDEV_FACTOR = float(8.0)
  slope = scipy.signal.gaussian(correlation.shape[0]/RAMP_SIZE_DIV, correlation.shape[0]/(RAMP_SIZE_DIV * RAMP_STDEV_FACTOR))
  flattop_filter = np.zeros(correlation.shape)
  flattop_filter[:len(slope)] += slope
  flattop_filter[-len(slope):] -= slope
  filter_cdf = 0
  for i in np.arange(flattop_filter.shape[0]):
    filter_cdf += flattop_filter[i]
    flattop_filter[i] = filter_cdf 
  correlation = correlation * flattop_filter

  # Silence edge effects
  correlation[:start + len(delta_filter)] = 0
  correlation[-(start + len(delta_filter)):] = 0

  # Final result
  candidate_peaks = scipy.signal.argrelmax(correlation, order=micro_filter_size)[0]
  highest_peak_height = np.max(correlation)

  # Somewhat arbitrarily select the first peak that is higher than 30% of the highest peak
  PEAK_CUTOFF_PERCENT = .3
  cutoff_height = PEAK_CUTOFF_PERCENT*highest_peak_height
  for peak_arg in candidate_peaks:
    if correlation[peak_arg] >= cutoff_height:
      result = peak_arg
      break

  # Final result in seconds
  result_time = sample_mids[result] / float(data.shape[0]) * song_duration
  print 'Expected drop: %f seconds'%result_time

  # Segment song
  before_drop = data[:sample_mids[result]]
  after_drop = data[sample_mids[result]:]

  if play_file:
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(before_drop.dtype.itemsize),channels=n_channels,rate=rate,output=True)
    stream.write(before_drop.tostring())

  PlotSignal(correlation, times)

  if play_file:
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(after_drop.dtype.itemsize),channels=n_channels,rate=rate,output=True)
    stream.write(after_drop.tostring())

#GetDrop('shortbloodred.wav')
#GetDrop('beam_mako.wav', True)
#GetDrop('overtime.wav', True)
#GetDrop('bloodred.wav', True)
#GetDrop('louder.wav')
#GetDrop('440hz.wav')
GetDrop('live_for_the_night.wav')
