import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import zipfile

# Load data from CSV file
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))  # Training data

# Extract relevant columns
X = train_data['CONTENT'].values  # Get the comment content
y = train_data['CLASS'].values  # Get the labels

# Feature Engineering: Extract text features from CONTENT using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # Use only text data from CONTENT
X_TFIDF = tfidf.fit_transform(X).toarray()  # Convert CONTENT to numerical features

# Initialize Stratified K-Fold
n_splits = 10  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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

# Perform Stratified K-Fold Cross-Validation
fold = 1
accuracies = []

for train_index, val_index in skf.split(X_TFIDF, y):
    print(f"Fold {fold}:")
    
    # Split data into training and validation sets
    X_train, X_val = X_TFIDF[train_index], X_TFIDF[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the validation set
    y_val_pred = model.predict(X_val)
    
    # Evaluate the model
    acc = accuracy_score(y_val, y_val_pred)
    accuracies.append(acc)
    print(f"Accuracy: {acc}")
    print(classification_report(y_val, y_val_pred))
    
    fold += 1

# Print average accuracy across folds
print(f"Average Accuracy: {sum(accuracies) / n_splits}")
