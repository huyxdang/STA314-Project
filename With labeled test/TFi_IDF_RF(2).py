import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import zipfile

n=3

# Load train and test data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')  # Update file path
train_data = pd.read_csv(z.open('train.csv'))
test_data = pd.read_csv(z.open('test.csv'))

# Prepare data
X_train = train_data['CONTENT'].values
Y_train = train_data['CLASS'].values
X_test = test_data['CONTENT'].values
test_ids = test_data['COMMENT_ID'].values

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n, n), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train final model and predict
final_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
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

