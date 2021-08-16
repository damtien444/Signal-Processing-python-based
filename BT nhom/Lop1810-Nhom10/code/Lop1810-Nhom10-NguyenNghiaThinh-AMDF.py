#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# find min of array x 
# input: x
# output: value min 'm'
def minx(x):
    m = x[0]
    for n in x:
        if(n < m): m = n
    return m


# find max of array x 
# input: x
# output: value max 'm'
def maxx(x):
    m = x[0]
    for n in x:
        if(n > m): m = n
    return m


# find sum of array x 
# input: x
# output: value sum 's'
def sumx(x):
    s = 0
    for n in x:
        s += n
    return s


# find mean of array x 
# input: x
# output: value mean of 's' with s is sum of array x
def meanx(x):
    s = 0
    if(len(x)==0): return 0
    for n in x:
        s +=n
    return s/len(x)


# Find the peak of a 'frame' of signal base on 'baseline'
# input: frame of signal
# output: a array contain low peaks of a 'frame' in signal called  'peakIndices'
def multi_peak_finding(frame, baseline):
    peakIndices = []
    peakIndex = None
    peakValue = None
    
    for index, value in enumerate(frame):
        if value < baseline:
            if peakValue == None or value < peakValue:
                peakIndex = index
                peakValue = value
        elif value > baseline and peakIndex != None:
            peakIndices.append(peakIndex)
            peakIndex = None
            peakValue = None
            
    if peakIndex != None:
        peakIndices.append(peakIndex)
        
    return peakIndices


# find the mean of signal to create an 'baseline' for 'multi_peak_finding' function
# input: signal x
# output: the mean of signal x
def baseline(x):
    return meanx(x)


# find the mean amplitude deviation of each frame in signal
# input: a frame of signal
# output: an array of mean amplitude deviation of each frame in signal - afAmdf - after using ADMF 
def computeAmdf(frame):
    
    k = len(frame)
    
    if k <= 0:
        return 0
    
    # create a matrix with one row and k colum
    afAmdf = np.ones(k)
    
    # find mean amplitude deviation in frames and add that value into array afAmdf
    for i in range(0,k):
        afAmdf[i] = meanx(np.abs(frame[np.arange(0, k - 1 - i)] - frame[np.arange(i + 1, k)]))
    return (afAmdf)



# Pitch Tracking of signal using AMDF
# input: 
#       x: the signal x read from file, 
#       FrameLength: the lenght of an frame in signal  
#       Hoplenght: Hop length of an frame in signal
#       f_s: sample rate of signal read from file and f_s = 16000Hz
# output: 
#       f : an array sampling frequency (f0) in each frame of signal  
#       t: time stamps
def PitchTimeAmdf(x, FrameLength, HopLength, f_s):

    # from 80 Hz to 250 Hz: fequency of male can speak
    # from 120 Hz to 400 Hz: fequency of female can speak
    # from 75 Hz to 350 Hz: the average fequency of adults can speak
    f_min = 120
    f_max = 400
    
    
    # the number of frame in signal
    NumOfFrames = x.size // HopLength
    
    # compute time stamps
    t = (np.arange(0, NumOfFrames) * HopLength + (FrameLength / 2)) / f_s

    # create a matrix 'f' contain sampling frequency (f0) in each frame
    f = np.zeros(NumOfFrames)
    
    # the limited samples of male or female can speak in signal
    sample_max = f_s / f_min
    sample_min = f_s / f_max
    
    
    # find threshold of signal x to determine which frame is voice sound or unvoice sound 
    cout = 0
    thres = 0
    for n in x:
        if(n > 0):
            thres += n
            cout +=1
    thres = thres / cout
    
    # find f0 in each frame
    for n in range(0, NumOfFrames):
        
        # the first position of frame
        i_start = n * HopLength
        # the last position of frame
        i_stop = minx([len(x) - 1, i_start + FrameLength - 1])

        # return true i_start < i_stop and false if  i_start >= i_stop
        if(i_start >= i_stop):
            continue
        else:
            # create a frame
            x_tmp = x[np.arange(i_start, i_stop + 1)]
            
            # if the max pitch in frame < threshold of signal 
            thres_frame = maxx(x_tmp)
            if(thres_frame < thres):
                f[n] = 0
                continue
            
            # compute AMDF (average magitude difference funtion) to create an array of AMDF - 'afAmdf'
            afAmdf = computeAmdf(x_tmp)
            
        # create an array contain position of value of peaks in afAmdf 
        peak_array = multi_peak_finding(afAmdf, baseline(afAmdf))
        
        # create an array contain the value of peaks in afAmdf
        k = [afAmdf[peak] for peak in peak_array]
        
        # return f0 = 0 in frame n if 'multi_peak_finding' function can not define the peak in this fame
        if(len(k)<=2): 
            f[n]=0
            continue
            
        # find f0 in frame n   
        index = 1
        while(index < len(k)):
            # find min value of peak in frame 
            if(k[index] == minx(k[1:len(k)-1])):  
                # if min value of peak in frame > sample_max, 
                # i will plus that min value to 1000 an then it is not a min value.
                if(peak_array[index]>sample_max):
                    k[index] += 1000
                    index=1
                elif(peak_array[index]<sample_min):
                    f[n]=0
                else:
                    f[n] = f_s / peak_array[index]
                    break
            index+=1
            
    return (f, t)


# In[11]:


f_s, x = read('G:/NAM 3/XU LY TIN HIEU SO/BT nhom/TinHieuMau/studio_female.wav')
f, t = PitchTimeAmdf(x, 320, 160,  f_s)


# In[12]:


## vẽ đồ thị tín hiệu và amdf của studio_male.wav
get_ipython().run_line_magic('matplotlib', '')
plt.figure(figsize=(20,5))
plt.subplot(2,1,1)
plt.title("studio_female.wav")
plt.xlabel("time stamps")
plt.ylabel("Amplitude")
plt.plot(x)
plt.subplot(2,1,2)
plt.title("Pitch Tracking of signal of studio_female.wav using AMDF")
plt.xlabel("time stamps")
plt.ylabel("f0")
plt.ylim([0,500])
plt.plot(t, f, 'o')
plt.show()


# In[4]:


# ##vẽ đồ thị khi thay đổi f_min và f_max của studio_female.wav
# %matplotlib
# f, t = PitchTimeAmdf(x, 320, 80,  f_s, 80, 250)
# plt.figure(figsize=(20,5))
# plt.subplot(3,1,1)
# plt.title("Pitch Tracking of signal of studio_female.wav using AMDF with f_min= 80Hz, f_max = 250Hz")
# plt.xlabel("time stamps")
# plt.ylabel("f0")
# plt.ylim([0,300])
# plt.plot(t, f, 'o')
# plt.show()
# plt.subplot(3,1,2)
# f, t = PitchTimeAmdf(x, 320, 160,  f_s, 40, 500)
# plt.title("Pitch Tracking of signal of studio_female.wav using AMDF with f_min= 40Hz, f_max = 500Hz")
# plt.xlabel("time stamps")
# plt.ylabel("f0")
# plt.ylim([0,300])
# plt.plot(t, f, 'o')
# plt.show()
# plt.subplot(3,1,3)
# f, t = PitchTimeAmdf(x, 320, 320,  f_s, 160, 200)
# plt.title("Pitch Tracking of signal of studio_female.wav using AMDF with f_min= 120Hz, f_max = 200Hz")
# plt.xlabel("time stamps")
# plt.ylabel("f0")
# plt.ylim([0,300])
# plt.plot(t, f, 'o')
# plt.show()


# In[5]:


# ## vẽ đồ thị khi thay đổi FrameLength của studio_female.wav
# %matplotlib
# f, t = PitchTimeAmdf(x, 320, 160,  f_s)
# plt.figure(figsize=(20,5))
# plt.subplot(2,1,1)
# plt.title("Pitch Tracking of signal of studio_female.wav using AMDF with FrameLength= 320 samples")
# plt.xlabel("time stamps")
# plt.ylabel("f0")
# plt.ylim([0,500])
# plt.plot(t, f, 'o')
# plt.show()
# plt.subplot(2,1,2)
# f, t = PitchTimeAmdf(x, 620, 320,  f_s)
# plt.title("Pitch Tracking of signal of studio_female.wav using AMDF with FrameLength= 640 samples")
# plt.xlabel("time stamps")
# plt.ylabel("f0")
# plt.ylim([0,500])
# plt.plot(t, f, 'o')
# plt.show()


# In[6]:


# vẽ đồ thị khi thay đổi HopLength của studio_female.wav
# %matplotlib
# f, t = PitchTimeAmdf(x, 320, 80,  f_s)
# plt.figure(figsize=(20,5))
# plt.subplot(3,1,1)
# plt.title("Pitch Tracking of signal of studio_male.wav using AMDF with HopLength= 80 samples")
# plt.xlabel("time stamps")
# plt.ylabel("f0")
# plt.ylim([0,300])
# plt.plot(t, f, 'o')
# plt.show()
# plt.subplot(3,1,2)
# f, t = PitchTimeAmdf(x, 320, 160,  f_s)
# plt.title("Pitch Tracking of signal of studio_male.wav using AMDF with HopLength= 160 samples")
# plt.xlabel("time stamps")
# plt.ylabel("f0")
# plt.ylim([0,300])
# plt.plot(t, f, 'o')
# plt.show()
# plt.subplot(3,1,3)
# f, t = PitchTimeAmdf(x, 320, 320,  f_s)
# plt.title("Pitch Tracking of signal of studio_male.wav using AMDF with HopLength= 320 samples")
# plt.xlabel("time stamps")
# plt.ylabel("f0")
# plt.ylim([0,300])
# plt.plot(t, f, 'o')
# plt.show()


# In[7]:


# ## vẽ môt frame trong tín hiệu studio_female
# # ## lab_male
# # # y = computeAmdf(x[16960:17280]) 
# # ## lab_female
# # # y = computeAmdf(x[14960:15180])
# # ## studio_male
# # y = computeAmdf(x[34480:34800]) 
# # ## studio_female
# y = computeAmdf(x[14432:14752]) 
# f_s = 16000
# f_min = 120
# f_max = 400
# def baseline(x):
#     return meanx(x)
# k1 = [peak for peak in multi_peak_finding(y, baseline(y))]
# k = [y[peak] for peak in k1]
# # plt.plot(k1, k, 'o')
# peak_array = multi_peak_finding(y, baseline(y))
# k = [y[peak] for peak in peak_array]
# f=0
# index=1
# while(index < len(k)):
#     if(k[index] == minx(k[1:len(k)-1])):
#         if(peak_array[index]>f_s/f_min): 
#             k.pop(index)
#             index=1
#         elif(peak_array[index]<f_s/f_max):
#             f = 0
#         else:
#             f = peak_array[index]
#             break
#     index+=1
    
# %matplotlib
# plt.figure(figsize=(20,10))
# plt.subplot(3,1,1)
# plt.title("Average Magnitude Diference Function of studio_female.wav")
# plt.plot(y)
# plt.subplot(3,1,2)
# plt.ylabel("Average Magnitude")
# plt.plot(y)
# plt.plot(k1, k, 'o')
# plt.subplot(3,1,3)
# plt.xlabel('samples')
# plt.plot(y)
# plt.plot(k1, k, 'o')
# plt.plot(f, k[index],'b*')           

