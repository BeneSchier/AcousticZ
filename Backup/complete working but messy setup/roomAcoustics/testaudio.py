import numpy as np
import soundfile as sf
from scipy import signal
import scipy
import sounddevice as sd

fs, audioIn = scipy.io.wavfile.read("C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/funnyantonia.wav")
# audioIn = audioIn[:, 0
# audioOut = scipy.signal.lfilter(self.ip, 1, audioIn)

# Example impulse response
RIR = np.zeros(5000)
RIR[0] = 1.0
RIR[1000] = 0.5
RIR[2000] = 0.25
RIR[3000] = 0.125

audioOut = scipy.signal.convolve(audioIn, RIR)
audioOut = np.real(audioOut)
audioOut = audioOut / np.max(audioOut)

T = 1000
sf.write("processed_audio.wav", audioOut, fs)
