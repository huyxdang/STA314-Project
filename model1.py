# Import packages 
import pandas as pd
import pickle
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as vectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data from CSV file
z = zipfile.ZipFile('local/file/path/raw_data_file.zip') # Put ur local path for ur ZIP file here
train_data = pd.read_csv(z.open('train.csv')) # Training data
test_data = pd.read_csv(z.open('test.csv')) # Test data

# Keep only relevant training columns
X_train = train_data['CONTENT'].values # Get just the comment content 
Y_train = train_data['CLASS'].values # Get the labels 

# Keep COMMENT_ID for test data but drop other irrelevant columns
test_comment_ids = test_data['COMMENT_ID'].values  # Save comment IDs for submission
X_test = test_data['CONTENT'].values # Get just the comment content

# Tokenization and Vectorization
# First fit on training data
tfidf_vect = vectorizer(use_idf=True, lowercase=True) 
X_train_tfidf = tfidf_vect.fit_transform(X_train)

# Then transform test data using the same vectorizer
X_test_tfidf = tfidf_vect.transform(X_test)

# Training the Model
model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)

# Make predictions on test data
Y_pred = model.predict(X_test_tfidf)

# Create submission dataframe
submission_df = pd.DataFrame({
    'COMMENT_ID': test_comment_ids,
    'CLASS': Y_pred
})

# Save submission file
submission_df.to_csv('submission1.csv', index=False)
