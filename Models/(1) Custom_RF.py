from feature_extractor import extract_text_features, create_feature_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import numpy as np
import psutil
import zipfile
import pandas as pd

# Load data, separate into training and testing X's and Y's
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))  # Training data
test_data = pd.read_csv(z.open('test.csv'))  # Test data
X_train = train_data['CONTENT'].values  # Get x-values of training data
Y_train = train_data['CLASS'].values  # Training data labels
X_test = test_data['CONTENT'].values  # Testing content
test_ids = test_data['COMMENT_ID'].values  # Comment IDs for submission

# Extract custom features
feature_matrix, feature_names = create_feature_matrix(X_train)  # Training data
feature_matrix_test, feature_names_test = create_feature_matrix(X_test)  # Testing data

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
    model = RandomForestClassifier(random_state=42)
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
average_metrics = metrics_df.mean(axis=0).to_dict()
print("\nAverage Metrics Across Folds:")
for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")