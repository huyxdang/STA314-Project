import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import zipfile

# Load data from CSV file
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))  # Training data
test_data = pd.read_csv(z.open('test.csv'))  # Test data

# Keep only relevant training columns
X_train = train_data['CONTENT'].values  # Get just the comment content 
Y_train = train_data['CLASS'].values  # Get the labels 

# Keep COMMENT_ID for test data but drop other irrelevant columns
test_ids = test_data['COMMENT_ID'].values  # Save comment IDs for submission
X_test = test_data['CONTENT'].values  # Get just the comment content

# Feature Engineering: Extract text features from CONTENT using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # Use only text data from CONTENT
X_train_TFIDF = tfidf.fit_transform(X_train).toarray()  # Convert CONTENT to numerical features
X_test_TFIDF = tfidf.transform(X_test).toarray()  # Apply the same transformation to test data

# XGBoost Classifier
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
model.fit(X_train_TFIDF, Y_train)

# Make predictions
Y_pred = model.predict(X_test_TFIDF)  # Use transformed test data

# Create submission dataframe
submission_df = pd.DataFrame({
    'COMMENT_ID': test_ids,
    'CLASS': Y_pred
})

# Save submission file
submission_df.to_csv('submissionXGB.csv', index=False)
