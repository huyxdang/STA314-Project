import pandas as pd
import numpy as np
import time
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

# Set your n-value here:
n = 3 #Best value, got 94%

# Load data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip') # Change file path
train_data = pd.read_csv(z.open('train.csv'))  # Training data
test_data = pd.read_csv(z.open('test.csv'))  # Test data

# Extract features and labels
X_train = train_data['CONTENT'].values  # Text content for training
Y_train = train_data['CLASS'].values  # Labels for training
X_test = test_data['CONTENT'].values  # Text content for testing
test_ids = test_data['COMMENT_ID'].values  # Comment IDs for the test data

# Create TF-IDF representation with character 6-grams
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n, n), max_features=5000)  # Character 6-grams
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the final model on the entire training dataset and make predictions on the test dataset
final_model = SVC(kernel='linear', probability=True, random_state=42)
final_model.fit(X_train_tfidf, Y_train)
Y_test_pred = final_model.predict(X_test_tfidf)

# Create predictions DataFrame
submission_df = pd.DataFrame({'COMMENT_ID': test_ids, 'CLASS_pred': Y_test_pred})

# Load ground truth labels for test data
test_with_labels = pd.read_csv('/Users/huydang/Desktop/STA314-Project/test_with_labels.csv')  # Update path if necessary

# Ensure ground truth labels have the correct columns
test_with_labels.rename(columns={'CLASS': 'CLASS_true'}, inplace=True)
test_with_labels = test_with_labels[['COMMENT_ID', 'CLASS_true']]

# Set COMMENT_ID as the index for both DataFrames
submission_df = submission_df.set_index('COMMENT_ID')
test_with_labels = test_with_labels.set_index('COMMENT_ID')

# Compare predictions to true labels
comparison_df = submission_df.merge(test_with_labels, left_index=True, right_index=True)

# Calculate accuracy and F1 score
accuracy = accuracy_score(comparison_df['CLASS_true'], comparison_df['CLASS_pred'])
f1 = f1_score(comparison_df['CLASS_true'], comparison_df['CLASS_pred'], average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

