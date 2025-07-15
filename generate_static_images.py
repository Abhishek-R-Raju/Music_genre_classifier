import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from music_genre_classification import load_data, prepare_datasets

def generate_confusion_matrix():
    """Generate and save confusion matrix for the web application."""
    # Create directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    print("Loading data...")
    data_path = "Data/genres_original"
    data = load_data(data_path)
    
    print("Preparing datasets...")
    X_train, X_test, y_train, y_test = prepare_datasets(data)
    
    print("Loading model...")
    model = tf.keras.models.load_model("best_model.h5")
    
    print("Predicting on test set...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data["mapping"], 
                yticklabels=data["mapping"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    print("Saving confusion matrix image...")
    plt.savefig('static/images/confusion_matrix.png')
    print("Confusion matrix saved to static/images/confusion_matrix.png")
    
    # Save genre mapping to file for the web app
    with open('genre_mapping.txt', 'w') as f:
        for genre in data["mapping"]:
            f.write(f"{genre}\n")
    print(f"Genre mapping saved to genre_mapping.txt")

if __name__ == "__main__":
    generate_confusion_matrix()