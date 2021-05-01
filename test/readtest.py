import librosa
import librosa.display
import matplotlib.pyplot as plt

audiopath = './test/input/blues.00000.wav'

y, sr = librosa.load(audiopath)
librosa.display.waveplot(y, sr=sr)

plt.show()
