# Import packages 
import pandas as pd
import pickle
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as vectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix, classification_report 

# Load data from CSV file
z = zipfile.ZipFile('youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))
print train_data

# Drop irrelevant columns; Keep only CONTENT and CLASS
ntrain_data = train_data.drop(["COMMENT_ID","AUTHOR","DATE","VIDEO_NAME"], axis=1, inplace=False) 
X = ntrain_data['CONTENT'].values # Get just the comment content 
Y = ntrain_data['CLASS'].values # Get the labels 

# Tokenization and Vectorization
tfidf_vect = vectorizer(use_idf=True, lowercase=True) 
X_train_tfidf = tfidf_vect.fit_transform(ntrain_data)
print(ntrain_data_tfidf.shape)






