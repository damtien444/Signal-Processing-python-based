import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

#divide signal into frames  
def divide_signal_into_frames(y, frame_size, frame_stride, fs):
    frame_len = int(fs*frame_size) # number of samples in a single frame
    frame_step = int(fs*frame_stride) # number of overlapping samples
    total_frames = int(np.ceil(float(np.abs(len(y)-frame_len))/frame_step))  # total frames of signal
    padded_y = np.append(np.array(y), np.zeros(frame_len * total_frames - len(y)))  #push new array to end of matrix array
    framed_y = np.zeros((total_frames, frame_len))  #frame y (shape)
    for i in range(total_frames): #loop through over of total frames
        framed_y[i] = padded_y[i*frame_step : i*frame_step + frame_len]  # Apply ag Xj * Xj+t
    return framed_y


#Compute Autocorr Of each frames
def calculate_difference(frame) :
    half_len_signal = len(frame)//2
    j = 0
    autocorr = np.zeros(half_len_signal) # Init new Empty array to contain autocorrs of frame
    for j in range(half_len_signal):
        for i in range(half_len_signal):
            diff = frame[i] - frame[i+j] # Do lech bien do
            autocorr[j] += diff**2 
    
    return autocorr 

#tau time autocorrrelation unit
#Tien Research
def normalize_with_cumulative_mean(autocorr, halflen):
    new_autocorr = autocorr
    new_autocorr[0] = 1
    running_sum = 0.0
    for tau in range(1,halflen):
        running_sum += autocorr[tau]
        new_autocorr[tau] = autocorr[tau]/((1/tau)*running_sum)
    
    return new_autocorr


def absolute_threshold(new_autocorr, halflen, threshold):
    #create new array with condition          
    temp = np.array(np.where(new_autocorr < threshold))
    if (temp.shape == (1,0)):
        tau = -1
    else : 
        tau = temp[:,0][0]
    return tau


# Tang do chinh xac cua cac dinh
# Noi suy parabol
def parabolic_interpolation(new_autocorr, tau, frame_len):
    if tau > 1 and tau < (frame_len//2-1):
        alpha = new_autocorr[tau-1]
        beta = new_autocorr[tau]
        gamma = new_autocorr[tau+1]
        improv = 0.5*(alpha - gamma)/(alpha - 2*beta + gamma)
    else :
        improv = 0
    
    new_tau = tau + improv
    #return new_tau
    return new_tau
    

def pitch_tracker(y, frame_size, frame_step, sr):
    #call to divide_signal and push args into it
    framed_y = divide_signal_into_frames(y, frame_size, frame_step, sr)
    pitches = []
    for i in range(len(framed_y)):
        #Find autocorrelation with i in rnage len of framed_y
        autocorr = calculate_difference(framed_y[i])
        #Find new autocorrelation with i in rnage len of framed_y
        new_autocorr = normalize_with_cumulative_mean(autocorr, frame_len//2)
        #find threshold
        tau = absolute_threshold(new_autocorr, frame_len//2, 0.16)
        new_tau = parabolic_interpolation(new_autocorr, tau, frame_len)
        if (new_tau == -1):
            pitch = 0
        else :
            pitch = sr/new_tau
        #push pitch into pitches array
        pitches.append(pitch)
    #get timestamps
    list_times = [int(i * frame_len / 2) for i in range(len(pitches))]
    #return a tuple type with pitches and times
    return (pitches, list_times)


if __name__ =='__main__': 
    sr, y = read('lab_female.wav')
    frame_size = 0.03
    frame_step = 0.025  #hop length
    frame_len = int(frame_size * sr) # #block length
    
    pitches = pitch_tracker(y, frame_size, frame_step, sr)
    #0 ->500
    plt.ylim([0, 500])
    #title
    plt.title('Pitch Tracking Of Signal lab_female.wav Using AutoCorrelation')
    #xlable
    plt.xlabel('Time Stamps')
    plt.ylabel('F0')
    # plot
    plt.plot(pitches[1], pitches[0], 'o')
    #show
    plt.show()

    