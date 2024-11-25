import pandas as pd
import numpy as np
import time
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Load data
z = '/Users/huydang/Desktop/STA314-Project/Full_data.csv'
train_data = pd.read_csv(z)  # Training data
test_data = pd.read_csv(z)  # Test data 

# Extract features and labels
X_train = train_data['CONTENT'].values  # Text content for training
Y_train = train_data['CLASS'].values  # Labels for training
X_test = test_data['CONTENT'].values  # Text content for testing
test_ids = test_data['COMMENT_ID'].values  # Comment IDs for the test data

# Create TF-IDF representation with word n-grams
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 6), max_features=5000)  # Word n-grams (1 to 2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []

# Training and validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_tfidf, Y_train)):
    print(f"Fold {fold + 1}")
    X_train_fold, X_val_fold = X_train_tfidf[train_idx], X_train_tfidf[val_idx]
    Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

    # Measure training time
    start_train_time = time.time()
    model = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=None,  # No maximum depth
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
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

# Train the final model on the entire training dataset and make predictions on the test dataset
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_train_tfidf, Y_train)
Y_test_pred = final_model.predict(X_test_tfidf)