#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import numpy as np
import librosa as lb
voice = pd.DataFrame()


# In[75]:


'''y = lb.load('available.wav',sr=None)
k = 44100
A= np.array(y)
B=np.array(A)
B'''


import soundfile as sf
filename = "bb.wav"
data, fs = sf.read(filename, dtype='float32')
print(data,fs)
lowcut=data.min()
print(lowcut)


# In[76]:


import numpy as np

def spectral_statistics(y: np.ndarray, fs: int, lowcut: int ) -> dict:
    """
    Compute selected statistical properties of spectrum
    :param y: 1-d signsl
    :param fs: sampling frequency [Hz]
    :param lowcut: lowest frequency [Hz]
    :return: spectral features (dict)
    """
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    idx = int(lowcut / fs * len(freq) * 2)
    spec = np.abs(spec[idx:])
    freq = freq[idx:]

    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

    top_peaks_ordered_by_power = {'stat_freq_peak_by_power_1': 0, 'stat_freq_peak_by_power_2': 0, 'stat_freq_peak_by_power_3': 0}
    top_peaks_ordered_by_order = {'stat_freq_peak_by_order_1': 0, 'stat_freq_peak_by_order_2': 0, 'stat_freq_peak_by_order_3': 0}
    amp_smooth = signal.medfilt(amp, kernel_size=15)
    peaks, height_d = signal.find_peaks(amp_smooth, distance=100, height=0.002)
    if peaks.size != 0:
        peak_f = freq[peaks]
        for peak, peak_name in zip(peak_f, top_peaks_ordered_by_order.keys()):
            top_peaks_ordered_by_order[peak_name] = peak

        idx_three_top_peaks = height_d['peak_heights'].argsort()[-3:][::-1]
        top_3_freq = peak_f[idx_three_top_peaks]
        for peak, peak_name in zip(top_3_freq, top_peaks_ordered_by_power.keys()):
            top_peaks_ordered_by_power[peak_name] = peak

    specprops = {
        'stat_mean': mean,
        'stat_sd': sd,
        'stat_median': median,
        'stat_mode': mode,
        'stat_Q25': Q25,
        'stat_Q75': Q75,
        'stat_IQR': IQR,
        'stat_skew': skew,
        'stat_kurt': kurt
    }
    specprops.update(top_peaks_ordered_by_power)
    specprops.update(top_peaks_ordered_by_order)
    return specprops



'''def spectral_properties(y, fs: int) -> dict:
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

    
    result_d = {
        'meanfreq': mean,
        'sd': sd,
        'median': median,
        'mode': mode,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt,
      
                    }
    return(result_d)'''


# In[77]:


dictionary = spectral_statistics(data,fs,lowcut)
print(dictionary)
voice= voice.append(dictionary, ignore_index=True)


# In[61]:


from scipy import signal
import numpy as np
import soundfile as sf

y, samplerate = sf.read('v2.wav') 
chunks = np.array_split(y,int(samplerate/2000))
peaks = []

for chunk in chunks:
    # simulated pure signal
    t = np.linspace(0, 1, samplerate)
    wave = chunk
    # compute the magnitude of the Fourier Transform and its corresponding frequency values
    freq_magnitudes = np.abs(np.fft.fft(wave))
    freq_values = np.fft.fftfreq(samplerate, 1/samplerate)
    # find the max. magnitude
    max_positive_freq_idx = np.argmax(freq_magnitudes[:samplerate//2 + 1])
    peaks.append(freq_values[max_positive_freq_idx])
meanfun = sum(peaks)/len(peaks)
print(meanfun)


# In[62]:



meanfun = sum(peaks)/len(peaks)
print(meanfun)


# In[63]:


print(voice)


# In[64]:


#voice.apply(lambda x: x/x.max(), axis=0)
voice.reset_index()


# In[65]:


voice.to_csv('test1.csv')


# In[66]:


k=pd.read_csv('test1.csv')
k.head()


# In[ ]:




