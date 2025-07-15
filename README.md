# Music Genre Classification Web Application

This web application uses deep learning to classify music into different genres and visualize audio features. It provides a user-friendly interface for uploading audio files (.mp3 or .wav) and displays the predicted genre along with various audio visualizations and model performance metrics.

## Overview

The application analyzes audio files using Mel-Frequency Cepstral Coefficients (MFCCs) and a Convolutional Neural Network (CNN) to predict the genre of music. It also generates various visualizations that help understand the audio features and model performance.

## Model Architecture and Performance

### CNN Architecture
The application uses a Convolutional Neural Network (CNN) with the following architecture:

- **Input Layer**: Takes MFCC features from audio segments
- **Convolutional Layers**:
  - First Conv Block: 32 filters (3×3), ReLU activation, BatchNormalization, MaxPooling (2×2), Dropout (0.25)
  - Second Conv Block: 64 filters (3×3), ReLU activation, BatchNormalization, MaxPooling (2×2), Dropout (0.25)
  - Third Conv Block: 128 filters (3×3), ReLU activation, BatchNormalization, MaxPooling (2×2), Dropout (0.25)
- **Dense Layers**:
  - Flatten layer
  - Dense layer with 512 units, ReLU activation, BatchNormalization, Dropout (0.3)
  - Dense layer with 256 units, ReLU activation, BatchNormalization, Dropout (0.3)
  - Output layer with 10 units (one per genre) and Softmax activation

### Model Performance
- **Test Accuracy**: 87.69%
- **Test Loss**: 0.3875

These metrics indicate that the model can correctly classify music genres with high accuracy.

### Confidence Score
When the model predicts a genre for an uploaded music file, it also provides a confidence score. This score represents the probability (as a percentage) that the model assigns to the predicted genre class. The higher the confidence score, the more certain the model is about its prediction.

## Files and Folders Structure

### Core Application Files

#### `app.py`
The main Flask web application that handles:
- HTTP routes and API endpoints
- File uploads and format conversion
- Serving predictions and visualizations
- Managing the web interface

#### `music_genre_classifier.py`
Core functionality for audio processing and machine learning:
- Audio loading and preprocessing
- MFCC feature extraction
- Genre prediction functions
- Audio visualization generation
- Confusion matrix generation

#### `music_genre_classification.py`
Original script that contains:
- The data loading and preprocessing pipeline
- CNN model architecture definition
- Model training and evaluation logic
- Original visualization functions
- This serves as the foundation for the web application

### Machine Learning Model Files

#### `best_model.h5`
- The trained Convolutional Neural Network model
- Contains the weights and architecture for genre prediction
- Used to make predictions on new audio files

#### `genre_mapping.txt`
- Maps numerical indices to genre names
- Created by `extract_genre_mapping.py`
- Critical for translating model outputs into human-readable genre labels

### Setup and Visualization Files

#### `extract_genre_mapping.py`
- Extracts genre names from dataset directories
- Creates a mapping file that translates numerical predictions to genre names
- Ensures consistent mapping between training and prediction

#### `generate_extended_visualizations.py`
- Generates advanced visualizations for the web interface
- Creates classification report, ROC curves, and MFCC analyses
- Pre-computes visualizations to improve web app performance

#### `generate_static_images.py`
- Generates the confusion matrix visualization
- Creates a static image file that can be served by the web application
- Helps visualize the model's performance across different genres

### Web Interface Files

#### `templates/index.html`
- Main HTML template for the web application
- Contains the file upload interface
- Displays prediction results and visualizations
- Implements tab-based navigation for different visualizations
- Shows model performance metrics in the footer section

#### `static/css/style.css`
- CSS styling for the web application
- Defines layout, colors, and responsive design
- Ensures a consistent and professional user experience

#### `static/js/main.js`
- Client-side JavaScript for interactivity
- Handles file uploads and form submission
- Updates the UI with prediction results and visualizations
- Manages tab switching and dynamic content

#### `static/images/`
- Directory containing pre-generated visualization images
- Includes confusion matrix, classification report, ROC curves
- Contains MFCC visualization analysis
- Stores mean and variance plots of audio features

### Data Folders

#### `Data/genres_original/`
- Contains the GTZAN dataset used for training
- Organized by genre subdirectories
- Holds .wav audio samples (30 seconds each)
- 10 genres with approximately 100 samples each

#### `uploads/`
- Temporary storage for user-uploaded audio files
- Stores both original uploads and converted files
- Cleaned periodically to manage disk space

## Application Functionality

### Audio Upload and Processing
1. Users upload MP3 or WAV audio files through the web interface
2. MP3 files are automatically converted to WAV format
3. Audio is preprocessed (resampling, segmentation) for feature extraction
4. MFCCs are extracted to represent the audio characteristics

### Genre Classification
1. The pre-trained CNN model analyzes the extracted features
2. The model predicts probabilities for each genre
3. The most likely genre is displayed with a confidence score
4. Confidence score calculation: `confidence = prediction[predicted_index] * 100`
5. Results are presented in an easy-to-understand format

### Audio Visualizations
The application generates and displays multiple visualizations:

1. **Waveform**: Time-domain representation of audio amplitude
2. **Spectrogram**: Frequency-time-amplitude representation
3. **Mel-Spectrogram**: Frequency representation mapped to the mel scale
4. **MFCC**: Mel-Frequency Cepstral Coefficients visualization

### Performance Metrics
The web app includes model performance visualizations:

1. **Confusion Matrix**: Shows prediction accuracy across genres
2. **Classification Report**: Displays precision, recall, and F1-score
3. **ROC Curves**: Illustrates model discrimination ability for each genre

### MFCC Analysis
Advanced visualizations for audio feature understanding:

1. **MFCC Heatmap**: Visualization of MFCC features for a sample
2. **Mean MFCCs**: Average MFCC pattern across the dataset
3. **MFCC Variance**: Variation in MFCC coefficients across samples

## Technical Details

### Audio Processing
- Sample Rate: 22,050 Hz
- Audio Duration: 30 seconds (for training data)
- Frame Size: 2,048 samples
- Hop Length: 512 samples
- MFCC Features: 13 coefficients
- Segments per Track: 10 (for training data)

### Neural Network Training
- Optimizer: Adam with learning rate 0.001
- Loss Function: Categorical Cross-Entropy
- Batch Size: 32
- Epochs: Up to 50 with early stopping
- Data Split: 80% training, 20% testing
- Regularization: Dropout and BatchNormalization
- Callbacks: 
  - EarlyStopping (monitors validation loss with patience of 5)
  - ModelCheckpoint (saves best model based on validation accuracy)
  - ReduceLROnPlateau (reduces learning rate when validation loss plateaus)

### Web Technologies
- Backend: Flask (Python)
- Frontend: HTML5, CSS3, JavaScript, Bootstrap
- Audio Processing: Librosa, Pydub
- Data Visualization: Matplotlib, Seaborn
- Machine Learning: TensorFlow, Keras

## Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

### Installation
1. Clone the repository
2. Install the required packages:
```
pip install -r requirements.txt
```

3. Generate visualizations (if not already present):
```
python generate_extended_visualizations.py
python generate_static_images.py
```

4. Start the web application:
```
python app.py
```

5. Open a web browser and navigate to:
```
http://127.0.0.1:5000
```

## Dataset

The application uses the GTZAN dataset, which includes:
- 1000 audio tracks (30 seconds each)
- 10 genres (Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock)
- 100 tracks per genre
- Audio files in .wav format (22050Hz Mono 16-bit)

## Model Training Process

1. **Data Preparation**:
   - Load audio files from the GTZAN dataset
   - Extract MFCC features from each audio file
   - Split each 30-second track into 10 segments
   - Create a balanced dataset with equal representation of genres

2. **Training**:
   - Split data into training (80%) and testing (20%) sets
   - Train the CNN model with early stopping to prevent overfitting
   - Save the best model based on validation accuracy

3. **Evaluation**:
   - Evaluate model performance on the test set
   - Generate confusion matrix and classification report
   - Analyze ROC curves for each genre

## Implementation Details

### Feature Extraction
- MFCC features capture the timbral characteristics of audio signals
- Each audio segment is transformed into a 2D MFCC feature matrix
- These features serve as the input to the CNN model

### MP3 to WAV Conversion
- The application uses Pydub library to convert MP3 files to WAV
- This ensures compatibility with the Librosa processing pipeline
- The converted files maintain the same audio quality

### Visualization Generation
- Audio visualizations are generated on-the-fly for uploaded files
- Model performance visualizations are pre-generated
- All visualizations use consistent color schemes and layouts for clarity

### Error Handling
- Robust error handling for audio file loading and processing
- Graceful degradation when encountering unsupported file formats
- Clear error messages for troubleshooting

## Future Improvements
- Real-time audio recording and classification
- Genre classification for specific sections of a song
- Enhanced visualization options
- Support for more audio formats
- Improved model accuracy through transfer learning
- Integration with music streaming platforms
- Batch processing of multiple files 