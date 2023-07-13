import numpy as np

def generate_monaural_RIR(V, c, t, fs, flow, fhigh, hist_data, delta_t):
    # Parameters
    num_bins = len(hist_data)
    num_samples = num_bins * int(delta_t * fs)
    num_frequencies = len(flow)
    num_channels = len(flow[0])
    
    # Compute mean event occurrence
    mu = (4 * np.pi * (c**3) * t**2) / (V)
    
    # Generate Dirac delta sequence
    def generate_dirac_deltas():
        deltas = np.zeros(num_samples)
        t0 = (3 / (2 * V)) * np.log(2) / (4 * np.pi * (c**3))
        max_length = num_bins * int(delta_t * fs)
        idx = 0
        while idx < max_length:
            z = np.random.rand()
            dtA = (1 / mu) * np.log(1 / z)
            if idx + int(dtA * fs) >= max_length:
                break
            deltas[idx + int(dtA * fs)] = 1
            idx += int(dtA * fs)
        return deltas
    
    dirac_deltas = generate_dirac_deltas()
    
    # Apply bandpass filtering
    def apply_bandpass_filter(input_signal, center_freq, low_freq, high_freq):
        output_signal = np.zeros_like(input_signal)
        for idx in range(num_samples):
            freq = idx * fs / num_samples
            if low_freq <= freq < high_freq:
                if freq < center_freq:
                    window = 0.5 * (1 + np.cos(2 * np.pi * freq / center_freq))
                else:
                    window = 0.5 * (1 - np.cos(2 * np.pi * freq / center_freq))
                output_signal[idx] = input_signal[idx] * window
        return output_signal
    
    bandpass_filtered = np.zeros((num_frequencies, num_channels, num_samples))
    
    for n in range(num_frequencies):
        for channel in range(num_channels):
            center_freq = flow[n][channel]
            low_freq = flow[n][channel] - (center_freq - flow[n-1][channel]) / 2
            high_freq = flow[n][channel] + (flow[n+1][channel] - center_freq) / 2
            
            bandpass_filtered[n][channel] = apply_bandpass_filter(dirac_deltas, center_freq, low_freq, high_freq)
    
    # Weighting with histogram data
    weighted_sequences = np.zeros(num_samples)
    
    for k in range(num_bins):
        g_k = int(k * delta_t * fs)
        vuut = 0
        for n in range(num_frequencies):
            for channel in range(num_channels):
                vuut += np.sum(bandpass_filtered[n][channel][g_k:g_k+int(delta_t*fs)]**2)
        
        for i in range(g_k, g_k+int(delta_t*fs)):
            vi = np.sum(bandpass_filtered[:, :, i]**2)
            En_k = hist_data[k]
            weighted_sequences[i] = vi * vuut / (En_k * np.sum(hist_data))
    
    # Construct monaural RIR
    monaural_RIR = np.zeros(num_samples)
    
    for n in range(num_frequencies):
        for channel in range(num_channels):
            monaural_RIR += bandpass_filtered[n][channel] * weighted_sequences
    
    return monaural_RIR

# Input parameters
V = 100  # Volume of the room
c = 343  # Speed of sound
t = 1.0  # Total duration
fs = 44100  # Sampling frequency
flow = [[1000], [2000]]  # Lower frequencies for each band/channel
fhigh = [[3000], [4000]]  # Upper frequencies for each band/channel
hist_data = [0.1, 0.2, 0.3, 0.2, 0.1]  # Energy histogram data
delta_t = 0.1  # Time slot for histogram data

# Generate monaural RIR
monaural_RIR = generate_monaural_RIR(V, c, t, fs, flow, fhigh, hist_data, delta_t)
