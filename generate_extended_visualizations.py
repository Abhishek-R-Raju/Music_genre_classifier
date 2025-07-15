import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.utils import to_categorical #type:ignore
from music_genre_classification import load_data, prepare_datasets
import tensorflow as tf

def generate_visualizations():
    """Generate and save additional visualizations for the web application."""
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
    
    # 1. Classification Report
    print("Generating classification report...")
    report = classification_report(y_test, y_pred_classes, target_names=data["mapping"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig('static/images/classification_report.png')
    plt.close()
    
    # 2. ROC Curve
    print("Generating ROC curve...")
    y_test_bin = to_categorical(y_test, num_classes=len(data["mapping"]))
    
    plt.figure(figsize=(10, 8))
    for i in range(len(data["mapping"])):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{data["mapping"][i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('static/images/roc_curve.png')
    plt.close()
    
    # 3. MFCC Heatmap of a sample
    print("Generating MFCC heatmap for a random sample...")
    idx = np.random.randint(0, len(data["mfcc"]))
    mfcc_sample = np.array(data["mfcc"][idx])
    label_idx = data["labels"][idx]
    genre = data["mapping"][label_idx]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(mfcc_sample.T, cmap='coolwarm')
    plt.title(f'MFCC Heatmap of Genre: {genre}')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficient Index')
    plt.tight_layout()
    plt.savefig(f'static/images/mfcc_heatmap_{genre}.png')
    plt.close()
    
    # 4. Mean and Variance of MFCCs across dataset
    print("Generating mean and variance of MFCCs across dataset...")
    all_mfccs = np.array(data["mfcc"])
    mean_mfcc = np.mean(all_mfccs, axis=0).T
    var_mfcc = np.var(all_mfccs, axis=0).T
    
    # Mean MFCC
    plt.figure(figsize=(12, 5))
    sns.heatmap(mean_mfcc, cmap="viridis")
    plt.title("Mean MFCCs Across Dataset")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficient")
    plt.tight_layout()
    plt.savefig('static/images/mean_mfcc.png')
    plt.close()
    
    # Variance MFCC
    plt.figure(figsize=(12, 5))
    sns.heatmap(var_mfcc, cmap="magma")
    plt.title("Variance of MFCCs Across Dataset")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficient")
    plt.tight_layout()
    plt.savefig('static/images/var_mfcc.png')
    plt.close()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    generate_visualizations() 