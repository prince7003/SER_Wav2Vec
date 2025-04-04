import librosa
import numpy as np
from config import config

def preprocess_audio(audio_file):
    audio, sr = librosa.load(audio_file, sr=config.sampling_rate)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    max_length = config.max_duration * config.sampling_rate
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        padding = max_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    return audio