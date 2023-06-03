import penn
import pydub
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from preproc import smooth


# convert mp3 file to wav
# path = '/home/martins_o/HackNote/pitch_follower_Slide-G.mp3'
# pydub.AudioSegment.from_mp3(path).export('/home/martins_o/HackNote/17573__danglada__g-major.wav', format="wav")
# exit()

# Load audio at the correct sample rate
audio = penn.load.audio('/home/martins_o/HackNote/334536__teddy_frost__piano-normal-d4.wav')[None, 0]

# audio = smooth(audio[0].numpy(), window_size=100)
#audio = audio.unsqueeze(0).float()


# How long is audio in seconds?
secs_in_audio = audio.shape[1] / penn.SAMPLE_RATE

# Here we'll use a 10 millisecond hopsize
hopsize = 0.01

# Provide a sensible frequency range given your domain and model
fmin = 30.
fmax = 1000.

# Choose a gpu index to use for inference. Set to None to use cpu.
gpu = 0

# If you are using a gpu, pick a batch size that doesn't cause memory errors
# on your gpu
batch_size = 2048

# Select a checkpoint to use for inference. The default checkpoint will
# download and use FCNF0++ pretrained on MDB-stem-synth and PTDB
checkpoint = penn.DEFAULT_CHECKPOINT

# Centers frames at hopsize / 2, 3 * hopsize / 2, 5 * hopsize / 2, ...
pad = True

# (Optional) Linearly interpolate unvoiced regions below periodicity threshold
interp_unvoiced_at = .065

# Infer pitch and periodicity
pitch, periodicity = penn.from_audio(
    audio,
    penn.SAMPLE_RATE,
    hopsize=hopsize,
    fmin=fmin,
    fmax=fmax,
    checkpoint=checkpoint,
    batch_size=batch_size,
    pad=pad,
    interp_unvoiced_at=interp_unvoiced_at,
    gpu=gpu)

# Plot pitch and periodicity with matplotlib x should be in seconds
# x label should be in seconds
fig1_name = 'pitch.png'
fig2_name = 'spectogram.png'

fig, axes = plt.subplots(2, 1, sharex=True)

# Plot the audio waveform
axes[0].plot(audio[0].detach().numpy())
axes[0].set_ylabel('Amplitude')

# # Plot the pitch values
time = np.linspace(0, audio.shape[1], pitch.shape[1])
axes[1].plot(time, pitch[0].cpu().numpy())
axes[1].set_ylabel('Pitch')
axes[1].set_xlabel('Time (s)')
plt.savefig(fig1_name)

# Plot the spectrogram
fig2, ax2 = plt.subplots(1, 1)
ax2.specgram(audio[0].detach().numpy(), Fs=penn.SAMPLE_RATE)
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Time (s)')
plt.savefig(fig2_name)

fig1_im = cv.imread(fig1_name)
fig2_im = cv.imread(fig2_name)

comb = np.vstack((fig1_im, fig2_im))
cv.imwrite('combined.png', comb)






