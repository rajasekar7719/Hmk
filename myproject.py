#  Audio Project - corrected version

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.spatial.distance import cosine
import os

# Option 1: files are in the same folder as this .py file
file1 =r"C:/Users/DELL/Desktop/my project.py/Voice1.wav.wav" 
file2 =r"C:/Users/DELL/Desktop/my project.py/Voice2.wav.wav" 


# Check if files exist
if not os.path.exists(file1):
    print(f"ERROR: File not found → {file1}")
    print("Current folder Python is looking in:", os.getcwd())
    exit()

if not os.path.exists(file2):
    print(f"ERROR: File not found → {file2}")
    print("Current folder Python is looking in:", os.getcwd())
    exit()

print("Both files found ✓")

# Load audio files
sr1, data1 = wavfile.read(file1)
sr2, data2 = wavfile.read(file2)

print(f"Voice 1: {len(data1)} samples, sample rate = {sr1} Hz")
print(f"Voice 2: {len(data2)} samples, sample rate = {sr2} Hz")

# Make both audios the same length (trim the longer one)
min_length = min(len(data1), len(data2))
data1 = data1[:min_length]
data2 = data2[:min_length]
print(f"Both trimmed to {min_length} samples")

# Convert stereo to mono if needed
if data1.ndim > 1:
    data1 = data1.mean(axis=1)
    print("Voice 1: converted stereo → mono")

if data2.ndim > 1:
    data2 = data2.mean(axis=1)
    print("Voice 2: converted stereo → mono")

# Compute spectrograms
f1, t1, S1 = spectrogram(data1, sr1, nperseg=1024)
f2, t2, S2 = spectrogram(data2, sr2, nperseg=1024)

print("Spectrograms created ✓")

# Plot side by side
plt.figure(figsize=(14, 6))

# Voice 1
plt.subplot(1, 2, 1)
plt.pcolormesh(t1, f1, 10 * np.log10(S1 + 1e-10), shading='gouraud', cmap='magma')
plt.title("Voice 1 - Spectrogram")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.colorbar(label="Power (dB)")

# Voice 2
plt.subplot(1, 2, 2)
plt.pcolormesh(t2, f2, 10 * np.log10(S2 + 1e-10), shading='gouraud', cmap='magma')
plt.title("Voice 2 - Spectrogram")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.colorbar(label="Power (dB)")

plt.tight_layout()
plt.savefig("voice_comparison.png", dpi=150)
print("Graph saved as: voice_comparison.png")

plt.show()

# --- Similarity Score ---
# Flatten log-spectrograms and compare
log_spec1 = np.log10(S1 + 1e-10).flatten()
log_spec2 = np.log10(S2 + 1e-10).flatten()

# Make them same length
min_len = min(len(log_spec1), len(log_spec2))
log_spec1 = log_spec1[:min_len]
log_spec2 = log_spec2[:min_len]

# Cosine similarity (1 = identical, 0 = very different, can be negative)
similarity = 1 - cosine(log_spec1, log_spec2)
similarity_percent = round(similarity * 100, 1)

print(f"\nSimilarity between the two voices: {similarity_percent}%")
print("=== Project Completed ===")
