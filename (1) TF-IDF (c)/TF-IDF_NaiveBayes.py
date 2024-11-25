import pandas as pd
import numpy as np
import zipfile
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

# Set your n-value here:
n = 8

# Load data
# Load data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))  # Training data
test_data = pd.read_csv(z.open('test.csv'))  # Test data


# Split training data into features and labels
X = train_data['CONTENT'].values  # Text content
Y = train_data['CLASS'].values  # Labels

# Create TF-IDF representation with character 6-grams
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, n), max_features=5000)  # Character 6-grams
X_tfidf = vectorizer.fit_transform(X)

# Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

# Training and validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, Y)):
    print(f"Fold {fold + 1}")
    X_train_fold, X_val_fold = X_tfidf[train_idx], X_tfidf[val_idx]
    Y_train_fold, Y_val_fold = Y[train_idx], Y[val_idx]

    # Measure training time
    start_train_time = time.time()
    model = MultinomialNB()
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

# Train the final model on the entire dataset
final_model = MultinomialNB()
final_model.fit(X_tfidf, Y)

# Save the final trained model
import joblib
joblib.dump(final_model, 'naive_bayes_tfidf_char6.pkl')

print("\nModel training complete. Final model saved as 'naive_bayes_tfidf_char6.pkl'.")
