import csv
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn
import os
'''
读取csv音频文件并输出音频信息
data[i,0] = 过零率(Zero Crossing Rate)
data[i,1] = y_harm_mean谐波均值
data[i,2] = y_harm_var谐波方差
data[i,3] = y_perc_mean感知激波均值
data[i,4] = y_perc_var感知激波方差
data[i,5] = Tempo BMP (beats per minute)（音乐节拍）
data[i,6] = spectral_centroids_mean光谱质心均值
data[i,7] = spectral_centroids_var光谱质心方差
data[i,8] = spectral_rolloff_mean光谱衰减均值
data[i,9] = spectral_rolloff_var光谱衰减方差
data[i,10~29] = mfccs_mean#Mel-Frequency Cepstral Coefficients(梅尔频率倒谱系数)均值
data[i,30~49] = mfccs_var#Mel-Frequency Cepstral Coefficients(梅尔频率倒谱系数)方差
data[i,50~61] = chromagram_mean色度频率均值
data[i,62~73] = chromagram_var色度频率方差
data[i,74] = chroma_stft_mean频率均值
data[i,75] = chroma_stft_var频率方差
data[i,76] = number
data[i,77] = 编号
'''

#load music
#audio_path = 'C:/Users/73714/code/sound_Recognition/music_wav/'
#y,sr = librosa.load(audio_path)   #输出为采样率为22050的单声道从0.0秒开始


data_path = 'C:/Users/73714/code/sound_Recognition/music_data/blues00000.csv'
video = np.loadtxt(open(data_path,"rb"),delimiter=",",skiprows=0)
sr = int(video[0])
y = video[1:]
audio_file, _ = librosa.effects.trim(y)

data = np.zeros(shape = (1,76))
i=0

#------------------------

#过零率(Zero Crossing Rate)

zero_crossings = librosa.zero_crossings(audio_file, pad=False)
#print(zero_crossings)
data[i,0] = sum(zero_crossings)

#------------------------
#Harmonics and Perceptrual
y_harm, y_perc = librosa.effects.hpss(audio_file)
#print(y_harm)#谐波
#print(y_perc)#感知激波
y_harm_mean = np.mean(y_harm)
y_harm_var = np.var(y_harm)
y_perc_mean = np.mean(y_perc)
y_perc_var = np.var(y_perc)

data[i,1] = y_harm_mean
data[i,2] = y_harm_var
data[i,3] = y_perc_mean
data[i,4] = y_perc_var

#------------------------

#Tempo BMP (beats per minute)（音乐节拍）
tempo, _ = librosa.beat.beat_track(y, sr = sr)
#print(tempo)
data[i,5] = tempo

#------------------------

#Spectral Centroid(光谱质心)
# Calculate the Spectral Centroids
spectral_centroids = librosa.feature.spectral_centroid(audio_file, sr=sr)[0]
spectral_centroids_mean = np.mean(spectral_centroids)
spectral_centroids_var = np.var(spectral_centroids)
# Shape is a vector
#print('Centroids:', spectral_centroids, '\n')
#print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')
#print(spectral_centroids_mean)
#print(spectral_centroids_var)

data[i,6] = spectral_centroids_mean
data[i,7] = spectral_centroids_var



# Computing the time variable for visualization
frames = range(len(spectral_centroids))

# Converts frame counts to time (seconds)
t = librosa.frames_to_time(frames)
#print('frames:', frames, '\n')
#print('t:', t)

# Function that normalizes the Sound Data
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#-----------------------

#Spectral Rolloff(光谱衰减)
# Spectral RollOff Vector
spectral_rolloff = librosa.feature.spectral_rolloff(audio_file, sr=sr)[0]
#print(spectral_rolloff)
spectral_rolloff_mean = np.mean(spectral_rolloff)
spectral_rolloff_var = np.var(spectral_rolloff)

data[i,8] = spectral_rolloff_mean
data[i,9] = spectral_rolloff_var

# The plot
#plt.figure(figsize = (16, 6))
#librosa.display.waveplot(audio_file, sr=sr, alpha=0.4, color = '#A300F9')
#plt.plot(t, normalize(spectral_rolloff), color='#FFB100')

#-----------------------

#Mel-Frequency Cepstral Coefficients(梅尔频率倒谱系数)
mfccs = librosa.feature.mfcc(audio_file, sr=sr)
#print('mfccs shape:', (np.shape(mfccs))
mfccs_mean = np.mean(mfccs,axis=1)
mfccs_var = np.var(mfccs,axis=1)
#print(np.shape(mfccs_mean))

for mfccs_i in range(20):
    data[i,10+mfccs_i] = mfccs_mean[mfccs_i]
    data[i,30+mfccs_i] = mfccs_var[mfccs_i]

'''
#Displaying  the MFCCs:
plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool')

# Perform Feature Scaling
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print('Mean:', mfccs.mean(), '\n')
print('Var:', mfccs.var())

plt.figure(figsize = (16, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool')
'''
#------------------------

#Chroma Frequencies(色度频率)
# Increase or decrease hop_length to change how granular you want your data to be
hop_length = 5000

# Chromogram
chromagram = librosa.feature.chroma_stft(audio_file, sr=sr, hop_length=hop_length)
#print('Chromogram shape:', chromagram.shape)

chromagram_mean = np.mean(chromagram,axis=1)
chromagram_var = np.var(chromagram,axis=1)

for chromagram_i in range(12):
    data[i,50+chromagram_i] = chromagram_mean[chromagram_i]
    data[i,62+chromagram_i] = chromagram_var[chromagram_i]

#plt.figure(figsize=(16, 6))
#librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

#----------------------

#Fourier Transform(傅里叶变换)
# Default FFT window size
n_fft = 2048                    # FFT window size
hop_length = 512                # number audio of frames between STFT columns (looks like a good default)
# Short-time Fourier transform (STFT)
D = np.abs(librosa.stft(audio_file, n_fft = n_fft, hop_length = hop_length))    #振幅D=（f，t）
frequent_weights  = D.sum(axis=1)               #各频率权重时间和（行为频率列为时间）
frequent_list = librosa.fft_frequencies(sr=sr, n_fft = n_fft)/sr
#print(np.shape(frequent_list))
#print(np.shape(frequent_weight))
#print(frequent_list)
#print(frequent_weight)
chroma_stft_mean = np.average(frequent_list,weights=frequent_weights)
chroma_stft_var = sum(((frequent_list-chroma_stft_mean)*frequent_weights)**2)/sum(frequent_weights)
#print(chroma_stft_mean)
#print(chroma_stft_var)

data[i,74] = chroma_stft_mean
data[i,75] = chroma_stft_var

print(np.shape(data))