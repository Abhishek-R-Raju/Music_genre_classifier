from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import tempfile
from werkzeug.utils import secure_filename
import tensorflow as tf
from music_genre_classifier import predict_genre, visualize_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = tf.keras.models.load_model("best_model.h5")

# Load genre mapping
with open('genre_mapping.txt', 'r') as f:
    genre_mapping = f.read().splitlines()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 file to WAV format."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting MP3 to WAV: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # If the file is mp3, convert to wav
        if filename.endswith('.mp3'):
            wav_filename = filename.rsplit('.', 1)[0] + '.wav'
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
            converted_path = convert_mp3_to_wav(file_path, wav_path)
            if converted_path:
                file_path = converted_path
            else:
                return jsonify({'error': 'Failed to convert MP3 to WAV format'})
        
        # Make prediction
        prediction = predict_genre(file_path, model, genre_mapping)
        
        # Generate visualizations
        visualization_data = visualize_features(file_path)
        
        return jsonify({
            'prediction': prediction,
            'waveform': visualization_data['waveform'],
            'spectrogram': visualization_data['spectrogram'],
            'mel_spectrogram': visualization_data['mel_spectrogram'],
            'mfcc': visualization_data['mfcc']
        })
    
    return jsonify({'error': 'Invalid file type. Please upload a .mp3 or .wav file.'})

@app.route('/confusion_matrix')
def get_confusion_matrix():
    img_path = 'static/images/confusion_matrix.png'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Confusion matrix image not found'})

@app.route('/classification_report')
def get_classification_report():
    img_path = 'static/images/classification_report.png'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Classification report image not found'})

@app.route('/roc_curve')
def get_roc_curve():
    img_path = 'static/images/roc_curve.png'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({'error': 'ROC curve image not found'})

@app.route('/mfcc_heatmap')
def get_mfcc_heatmap():
    # Get the first file that matches the pattern
    for file in os.listdir('static/images'):
        if file.startswith('mfcc_heatmap_'):
            img_path = os.path.join('static/images', file)
            return send_file(img_path, mimetype='image/png')
    
    return jsonify({'error': 'MFCC heatmap image not found'})

@app.route('/mean_mfcc')
def get_mean_mfcc():
    img_path = 'static/images/mean_mfcc.png'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Mean MFCC image not found'})

@app.route('/var_mfcc')
def get_var_mfcc():
    img_path = 'static/images/var_mfcc.png'
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Variance MFCC image not found'})

@app.route('/performance')
def get_performance():
    # Return model performance metrics
    performance_data = {
        'accuracy': 0.85,  # Example value
        'precision': 0.83,
        'recall': 0.82,
        'f1_score': 0.82
    }
    return jsonify(performance_data)

if __name__ == '__main__':
    app.run(debug=True) 