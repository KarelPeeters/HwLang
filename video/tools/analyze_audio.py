import json
import sys

import librosa
import numpy as np

path, out = sys.argv[1], sys.argv[2]
y, sr = librosa.load(path, sr=22050, mono=True)
hop = 512
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="time")
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop, units="time")
rms = librosa.feature.rms(y=y, hop_length=hop)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

# band energies for visuals (low / mid / high), downsampled to 60 fps
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
def band(lo, hi):
    return S[(freqs >= lo) & (freqs < hi)].mean(axis=0)
bands = {"low": band(20, 150), "mid": band(150, 2000), "high": band(2000, 11000), "rms": rms}
fps = 60
t60 = np.arange(0, times[-1], 1 / fps)
env = {}
for k, v in bands.items():
    v = np.interp(t60, times, v)
    v = v / (np.percentile(v, 99) + 1e-9)
    env[k] = np.round(np.clip(v, 0, 1.5), 4).tolist()

# structural segmentation via recurrence on chroma + mfcc
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=13)
feat = np.vstack([librosa.util.normalize(chroma, axis=1), librosa.util.normalize(mfcc, axis=1)])
bounds = librosa.segment.agglomerative(feat, 24)
bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop)

json.dump({
    "duration": float(len(y) / sr),
    "tempo": float(np.atleast_1d(tempo)[0]),
    "beats": [round(float(b), 4) for b in beats],
    "onsets": [round(float(o), 4) for o in onsets],
    "segments": [round(float(b), 3) for b in bound_times],
    "env_fps": fps,
    "env": env,
}, open(out, "w"))
print("tempo", tempo, "beats", len(beats), "segments", [round(float(b), 1) for b in bound_times])
