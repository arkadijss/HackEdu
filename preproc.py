import librosa
import numpy as np
from scipy.signal import lfilter

# Load audio file
audio_file = '/home/martins_o/HackNote/334536__teddy_frost__piano-normal-d4.wav'
y, sr = librosa.load(audio_file, sr=8000)

def smooth(y, window_size=100):
    # Calculate amplitude envelope
    amplitude_envelope = np.abs(y)

    # Apply smoothing filter
    smoothed_envelope = lfilter(np.ones(window_size)/window_size, 1, amplitude_envelope)

    # Normalize the smoothed envelope
    normalized_envelope = smoothed_envelope / np.max(smoothed_envelope)

    # Apply the smoothed envelope to the original signal
    y_smoothed = y * normalized_envelope
    return y_smoothed

    # Perform other pre-processing steps and pitch detection on the smoothed signal
# Perform pitch detection on each frame
# Use your pitch detection algorithm here

# Process the detected pitches as needed