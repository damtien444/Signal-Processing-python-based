#!/usr/bin/env python
# coding: utf-8

# In[1]:


def find_baseline(data, sampling_rate, PRODUCTS):

    '''
    
    To tell the difference of the non-voice and voice part of the signal
    we have to form a baseline based on the power spectrum.
    
    This function assume the first 2000 samples is in the non-voice part of the signal.
    It'll use the Harmonic Product Spectrum function to generate the corresponding baseline.
    
    Input: 
        data : array-like            
                        Full-length signal.
        sampling_rate : int  
                        Sampling rate of the signal.
        PRODUCTS : int 
                        Number of products to evaluate the harmonic product spectrum over.
    
    Output:
        maxv : int 
                        Base energy level that the algorithm based on to filter out the non-voice part's f0.
    
    '''
    
    WINDOW_SIZE = 2000
    window = np.ones(WINDOW_SIZE)
    i=1
    get_data_slice = data[int((0.5*i-0.5)*WINDOW_SIZE) : int((0.5*i+0.5)*WINDOW_SIZE)]

    fourier_transform = np.fft.rfft(get_data_slice*window, 2**13)

    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)

    down = []
    for i in range(1, PRODUCTS+1):
        down.append(power_spectrum[::i])
    #     down1 = power_spectrum[::1]

    res = []
    for i in range(len(down[-1])):
        mul = 1
        for j in range(0, PRODUCTS):
            mul *= down[j][i]
        res.append(mul)
    
    maxv = max(res)
    
    return maxv


# In[2]:


import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from scipy.signal import resample

get_ipython().run_line_magic('matplotlib', 'tk')

'''
    Harmonic product spectrum.

    This algorithm is used for fundamental frequency detection. It evaluates
    the magnitude of the FFT (fast Fourier transform) of the input signal, keeping
    only the positive frequencies. It then element-wise-multiplies this spectrum
    by the same spectrum downsampled by 2, then 3, ..., finally ending after
    numProd downsample-multiply steps.

    Parameters
    ----------
    FILE_PATH : String
      Specify where to locate original signal
      
    PRODUCTS : int
      Number of products to evaluate the harmonic product spectrum.
      
    FFT_POINT : int
      The length of the FFT. FFT lengths with low prime factors (i.e., products of 2,
      3, 5, 7) are usually (much) faster than those with high prime factors.
      
    LOW_PASS : int
      Lowest frequency that will be in the output
      
    HIGH_PASS : int
      Highest frequency that will be in the output
      
    WINDOW_SIZE : int
      The size of window of signal
      
    Return
    ----------
    Result : Array-like
      List of fundamental frequency corresponding with each window
'''

def find_f0_using_HPS(WINDOW_SIZE, LOW_PASS, HIGH_PASS, PRODUCTS, FILE_PATH, FFT_POINT):

    # Read data from original file
    sampling_rate, data = wavfile.read(FILE_PATH)
    # Innitiallize hanning window with the same size ofthe window
    window = np.hanning(WINDOW_SIZE)
    # Calculation of basic baseline to filter noise later
    base = find_baseline(data, sampling_rate, PRODUCTS)
    # Result list variable
    result = []

    # Slide through the signal with windows that overlaps 50% of each other
    for i in range(1, int(len(data)/(WINDOW_SIZE))*2-1):

        # EXTRACT DATA FROM SIGNAL AND COMPUTE FFT
        # Slice data with the windowsize that overlaps 50%
        get_data_slice = data[int((0.5*i-0.5)*WINDOW_SIZE) : int((0.5*i+0.5)*WINDOW_SIZE)]
        # Generate the fourier power and its corresponding frequency
        fourier_transform = np.fft.rfft(get_data_slice*window, FFT_POINT)
        frequency = np.linspace(0, sampling_rate/2, num=len(fourier_transform))

        # BANDPASS FILTERING
        # Create filter the value of too high or too low frequencies in the fourier result
        pas = []
        for i in range(len(frequency)):
            # Cut off the too low and too high frequencies
            if frequency[i]<LOW_PASS or frequency[i]>HIGH_PASS*5:
                pas.append(0)
            else:
                pas.append(1)
        # Apply the filter to the result of the FFT
        for i in range(len(frequency)):
            fourier_transform[i] *= pas[i]
        # Compute the Power Spectrum
        abs_fourier_transform = np.abs(fourier_transform)
        power_spectrum = np.square(abs_fourier_transform)

        # DOWNSAMPLING
        # Downsampling by the numnber of products
        down = [] # This is for saving the downsampled sequence
        for i in range(1, PRODUCTS+1):
            # Here, "downsampling a vector by N" means keeping only every N samples: downsample(v, N) = v[::N].
            down.append(power_spectrum[::i])

        # MULTIPLICATION
        # Element-wise-multiplies by the downsampled spectrum versions 
        res = [] # this is for saving the result after the multiplication
        for i in range(len(down[-1])):
            mul = 1
            for j in range(0, PRODUCTS):
                mul *= down[j][i]
            res.append(mul)

        # FIND THE PEAK AND EXTRACT F0
        # Find the index of the max peak of the multiplied result of the downsample versions
        maxv = max(res)
        indv = res.index(maxv)
        # Compare with the base TO FILTER OUT non_sound parts
        if maxv < base:
            indv = 0
        # Final compare with the condition then extract f0 for this frame    
        if frequency[indv] < LOW_PASS or frequency[indv] > HIGH_PASS:
            result.append(0)
            continue
        else:
            result.append(frequency[indv])
        
    return result


# In[3]:


import timeit

WINDOW_SIZE = 740

LOW_PASS = 20
HIGH_PASS = 200

# Declare the 
FFT_POINT = 2**13

PRODUCTS = 5

FILE_PATH = "studio_male.wav"
time_count = []

start = timeit.default_timer()
f0_list = find_f0_using_HPS(WINDOW_SIZE, LOW_PASS, HIGH_PASS, 3, FILE_PATH, 2**13)
end = timeit.default_timer()
time_count.append(end-start)

start = timeit.default_timer()
f0_list1 = find_f0_using_HPS(WINDOW_SIZE, LOW_PASS, HIGH_PASS, 5, FILE_PATH, 2**13)
end = timeit.default_timer()
time_count.append(end-start)

start = timeit.default_timer()
f0_list2 = find_f0_using_HPS(WINDOW_SIZE, LOW_PASS, HIGH_PASS, 7, FILE_PATH, 2**13 )
end = timeit.default_timer()
time_count.append(end-start)

start = timeit.default_timer()
f0_list3 = find_f0_using_HPS(WINDOW_SIZE, LOW_PASS, HIGH_PASS, 9, FILE_PATH, 2**13)
end = timeit.default_timer()
time_count.append(end-start)

time = []
for i in range(len(f0_list)):
    time.append(int(i*WINDOW_SIZE-WINDOW_SIZE/2))
    
time1 = []
for i in range(len(f0_list1)):
    time1.append(int(i*WINDOW_SIZE-WINDOW_SIZE/2))
    
time2 = []
for i in range(len(f0_list2)):
    time2.append(int(i*WINDOW_SIZE-WINDOW_SIZE/2))
    
time3 = []
for i in range(len(f0_list3)):
    time3.append(int(i*WINDOW_SIZE-WINDOW_SIZE/2))

fig, axs = plt.subplots(4, 1) 
axs[0].title.set_text("Pitch contour of lab_female.wav using HPS using PRODUCTS level of 3")
axs[0].set_ylabel("Fundamental Frequency")
axs[0].set_xlabel("Timestamps (samples)")
axs[0].plot(time, f0_list, 'o')

axs[1].title.set_text("Pitch contour of lab_female.wav using HPS using PRODUCTS level of 5")
axs[1].set_ylabel("Fundamental Frequency")
axs[1].set_xlabel("Timestamps (samples)")
axs[1].plot(time1, f0_list1, 'o')

axs[2].title.set_text("Pitch contour of lab_female.wav using HPS using PRODUCTS level of 10")
axs[2].set_ylabel("Fundamental Frequency")
axs[2].set_xlabel("Timestamps (samples)")
axs[2].plot(time2, f0_list2, 'o')

axs[3].title.set_text("Pitch contour of lab_female.wav using HPS using PRODUCTS level of 15")
axs[3].set_ylabel("Fundamental Frequency")
axs[3].set_xlabel("Timestamps (samples)")
axs[3].plot(time3, f0_list3, 'o')

# plt.plot(range(len(power_spectrum)), power_spectrum)
print(time_count)

