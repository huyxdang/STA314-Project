# Import packages 
import pandas as pd
import pickle
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as vectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report 

# Load data from CSV file
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv')) # Training data
test_data = pd.read_csv(z.open('test.csv')) # Test data

# Drop irrelevant columns for Testing dataset
ntest_data = test_data.drop(["COMMENT_ID","AUTHOR","DATE","VIDEO_NAME"], axis=1, inplace=False) 
X_test = ntest_data['CONTENT'].values # Get just the comment content 

# Tokenization and Vectorization
# First fit on training data
tfidf_vect = vectorizer(use_idf=True, lowercase=True) 
X_train_tfidf = tfidf_vect.fit_transform(X_train)

# Then transform test data using the same vectorizer
# Note: use transform() not fit_transform() for test data
X_test_tfidf = tfidf_vect.transform(X_test)

# Training the Model
model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)


