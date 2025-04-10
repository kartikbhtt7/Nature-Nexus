import librosa
import numpy as np

class AudioConfig:
    sr = 16000
    duration = 3
    hop_length = 340 * duration
    fmin = 20
    fmax = sr // 2
    n_mels = 128
    n_fft = 128 * 20
    samples = sr * duration

def preprocess_audio(audio_path, config=None):
    if config is None:
        config = AudioConfig()
        
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.sr)
    
    # Trim or pad
    if len(y) > config.samples:
        y = y[:config.samples]
    else:
        padding = config.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, padding - offset), 'constant')
    
    # Create mel spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=config.sr,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        fmin=config.fmin,
        fmax=config.fmax
    )
    spectrogram = librosa.power_to_db(spectrogram)
    
    # Return with correct shape for PyTorch (channels, height, width)
    return spectrogram[np.newaxis, ...]