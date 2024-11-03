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

# Drop irrelevant columns; Keep only CONTENT and CLASS
ntrain_data = train_data.drop(["COMMENT_ID","AUTHOR","DATE","VIDEO_NAME"], axis=1, inplace=False) 
X = ntrain_data['CONTENT'].values # Get just the comment content 
Y = ntrain_data['CLASS'].values # Get the labels 

# Tokenization and Vectorization
tfidf_vect = vectorizer(use_idf=True, lowercase=True) 
X_tfidf = tfidf_vect.fit_transform(X)
print("Original dataframe shape:", ntrain_data.shape)
print("Text content array shape:", X.shape)
print("TF-IDF matrix shape:", X_tfidf.shape)

# Training the Model 
model = MultinomialNB()
model.fit(X_tfidf, Y)






