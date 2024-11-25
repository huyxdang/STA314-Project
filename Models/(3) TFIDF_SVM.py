import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import psutil
import zipfile

# Load data
z = '/Users/huydang/Desktop/STA314-Project/Full_data.csv'
train_data = pd.read_csv(z)  # Training data
test_data = pd.read_csv(z)  # Test data

# Extract features and labels
X_train = train_data['CONTENT'].values  # Text content for training
Y_train = train_data['CLASS'].values  # Labels for training
X_test = test_data['CONTENT'].values  # Text content for testing
test_ids = test_data['COMMENT_ID'].values  # Comment IDs for the test data

# TF-IDF Feature Extraction
tfidf = TfidfVectorizer(max_features=5000)  # Limit features to 5000 for better performance
feature_matrix = tfidf.fit_transform(X_train).toarray()  # Transform training data to numerical features

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []

# Track memory usage
def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Training and validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(feature_matrix, Y_train)):
    print(f"Fold {fold + 1}")
    X_train_fold, X_val_fold = feature_matrix[train_idx], feature_matrix[val_idx]
    Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

    # Measure training time
    start_train_time = time.time()
    model = SVC(
        kernel='rbf',  # Radial Basis Function kernel
        C=1.0,         # Regularization parameter
        gamma='scale', # Kernel coefficient for RBF
        probability=True,  # Enable probability estimates
        random_state=42
    )
    model.fit(X_train_fold, Y_train_fold)
    train_time = time.time() - start_train_time

    # Measure inference time
    start_inference_time = time.time()
    Y_val_pred = model.predict(X_val_fold)
    inference_time = time.time() - start_inference_time

    # Calculate metrics
    accuracy = accuracy_score(Y_val_fold, Y_val_pred)
    precision = precision_score(Y_val_fold, Y_val_pred, average='weighted')
    recall = recall_score(Y_val_fold, Y_val_pred, average='weighted')
    f1 = f1_score(Y_val_fold, Y_val_pred, average='weighted')
    
    # AUC-ROC (requires probability predictions)
    Y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    auc_roc = roc_auc_score(Y_val_fold, Y_val_pred_proba)

    # Record metrics
    fold_metrics.append({
        "Fold": fold + 1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc,
        "Training Time (s)": train_time,
        "Inference Time (s)": inference_time,
        "RAM Usage (MB)": memory_usage()
    })

# Create a DataFrame to display metrics
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df)

# Average metrics across folds
average_metrics = metrics_df.drop(columns=["Fold"]).mean(axis=0).to_dict()
print("\nAverage Metrics Across Folds:")
for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")

