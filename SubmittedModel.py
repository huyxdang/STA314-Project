import pandas as pd
import numpy as np
import time
import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# This is the code for Random Forest, TF-IDF embedding, character n-gram with n = 5. It was chosen as the final submission for Kaggle. 

# Set your n-value here:
n = 5

# Load data
# Load data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip') # Change file path
train_data = pd.read_csv(z.open('train.csv'))  # Training data
test_data = pd.read_csv(z.open('test.csv'))  # Test data


# Split training data into features and labels
X = train_data['CONTENT'].values  # Text content
Y = train_data['CLASS'].values  # Labels

# Create TF-IDF representation with character n-gram
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n, n), max_features=5000) 
X_tfidf = vectorizer.fit_transform(X)

# Instantiate RF model
model = RandomForestClassifier(
    n_estimators=4000,  # Number of trees
    max_depth=None,  # No maximum depth
    n_jobs=-1  # Use all available cores
)

# Fit the model on the entire training dataset
model.fit(X_tfidf, Y)

# Transform the test data
X_test_tfidf = vectorizer.transform(test_data['CONTENT'].values)

# Predict labels for the test dataset
test_predictions = model.predict(X_test_tfidf)

# Create a submission DataFrame
submission_df = pd.DataFrame({
    "COMMENT_ID": test_data['COMMENT_ID'],  # Assuming 'ID' column is present in test data
    "CLASS": test_predictions
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
