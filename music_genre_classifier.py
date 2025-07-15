import os
import math
import numpy as np
import matplotlib
# Set the backend to 'Agg' for non-interactive mode
matplotlib.use('Agg')
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64
import tensorflow as tf

# Constants
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 10

def load_audio_file(file_path):
    """Load audio file with error handling."""
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        return signal, sr
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None

def extract_features(signal, sr):
    """Extract MFCC features from audio signal."""
    try:
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=NUM_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        return mfcc.T
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def predict_genre(file_path, model, genre_mapping):
    """Predict genre of audio file and return prediction."""
    signal, sr = load_audio_file(file_path)
    if signal is None:
        return "Error"

    # Trim the signal if it's too long
    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[:SAMPLES_PER_TRACK]
    
    # Pad if signal is too short
    if len(signal) < SAMPLES_PER_TRACK:
        signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')
    
    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    
    # Process first segment for prediction
    start = 0
    finish = samples_per_segment
    mfcc = extract_features(signal[start:finish], sr)
    
    if mfcc is not None:
        # Reshape the features to match the input shape of the model
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        
        # Make prediction
        prediction = model.predict(mfcc)[0]
        predicted_index = np.argmax(prediction)
        
        return genre_mapping[predicted_index]
    else:
        return "Error"

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def visualize_features(file_path):
    """Generate visualizations for audio file and return as base64 encoded images."""
    signal, sr = load_audio_file(file_path)
    if signal is None:
        return None
    
    # Trim if signal is too long
    if len(signal) > SAMPLES_PER_TRACK:
        signal = signal[:SAMPLES_PER_TRACK]
    
    # Create visualization dictionary
    visualization_data = {}
    
    # 1. Waveform
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sr, ax=ax)
    ax.set_title('Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    visualization_data['waveform'] = fig_to_base64(fig)
    
    # 2. Spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    visualization_data['spectrogram'] = fig_to_base64(fig)
    
    # 3. Mel-Spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title('Mel-Spectrogram')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    visualization_data['mel_spectrogram'] = fig_to_base64(fig)
    
    # 4. MFCC
    fig, ax = plt.subplots(figsize=(10, 4))
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    img = librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LENGTH, x_axis='time', ax=ax)
    ax.set_title('MFCC')
    fig.colorbar(img, ax=ax)
    visualization_data['mfcc'] = fig_to_base64(fig)
    
    return visualization_data

def generate_confusion_matrix(y_true, y_pred, genre_mapping):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genre_mapping, yticklabels=genre_mapping)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/confusion_matrix.png')
    plt.close()