import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Load data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))
test_data = pd.read_csv(z.open('test.csv'))

# Prepare training data
X_train = train_data['CONTENT'].values
y_train = train_data['CLASS'].values
test_comment_ids = test_data['COMMENT_ID'].values
X_test = test_data['CONTENT'].values

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,          # Limit vocabulary size
        min_df=2,                   # Ignore terms that appear in less than 2 documents
        max_df=0.95,                # Ignore terms that appear in more than 95% of documents
        ngram_range=(1, 2),         # Use both unigrams and bigrams
        stop_words='english'        # Remove common English words
    )),
    ('clf', MultinomialNB(alpha=1.0))  # Add smoothing
])

# Define parameter grid for GridSearchCV
param_grid = {
    'tfidf__max_features': [3000, 5000, 7000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__alpha': [0.1, 0.5, 1.0, 2.0]
}

# Perform k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Get cross-validation scores
cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

# Make predictions using the best model
Y_pred = grid_search.predict(X_test)

# Create submission file
submission_df = pd.DataFrame({
    'COMMENT_ID': test_comment_ids,
    'CLASS': Y_pred
})

submission_df.to_csv('submission2.csv', index=False)