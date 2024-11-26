import pandas as pd
import numpy as np
import time
import psutil
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
import zipfile

# Set your n-value here:
n = 7

# Load data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip') # Change file path
train_data = pd.read_csv(z.open('train.csv'))  # Training data
test_data = pd.read_csv(z.open('test.csv'))  # Test data

# Extract features and labels
X_train = train_data['CONTENT'].values  # Text content for training
Y_train = train_data['CLASS'].values  # Labels for training
X_test = test_data['CONTENT'].values  # Text content for testing
test_ids = test_data['COMMENT_ID'].values  # Comment IDs for the test data

# Create Bag of Words representation with character n-grams (length 6)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, n))  # Character 6-grams
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

"""
# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []

# Training and validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_bow, Y_train)):
    print(f"Fold {fold + 1}")
    X_train_fold, X_val_fold = X_train_bow[train_idx], X_train_bow[val_idx]
    Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

    # Measure training time
    start_train_time = time.time()
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train_fold, Y_train_fold)
    train_time = time.time() - start_train_time

    # Measure inference time
    start_inference_time = time.time()
    Y_val_pred = model.predict(X_val_fold)
    inference_time = time.time() - start_inference_time

    # Calculate metrics
    accuracy = accuracy_score(Y_val_fold, Y_val_pred)
    f1 = f1_score(Y_val_fold, Y_val_pred, average='weighted')

    # Record metrics
    fold_metrics.append({
        "Fold": fold + 1,
        "Accuracy": accuracy,
        "F1-Score": f1,
        "Inference Time (s)": inference_time
    })

# Create a DataFrame to display metrics
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df)

# Average metrics across folds
average_metrics = metrics_df.drop(columns=["Fold"]).mean(axis=0).to_dict()
print("\nAverage Metrics Across Folds:")
for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")
"""
# Train the final model on the entire training dataset and make predictions on the test dataset
final_model = SVC(kernel='linear', probability=True, random_state=42)
final_model.fit(X_train_bow, Y_train)
Y_test_pred = final_model.predict(X_test_bow)

# Create a submission DataFrame
submission_df = pd.DataFrame({
    'COMMENT_ID': test_ids,
    'CLASS': Y_test_pred
})

# Save submission DataFrame to a CSV file
submission_file = 'submission_SVC_BOW.csv'
submission_df.to_csv(submission_file, index=False)

print(f"Submission file saved as {submission_file}")
