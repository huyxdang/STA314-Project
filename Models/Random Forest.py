from feature_extractor import extract_text_features, create_feature_matrix
from data_functions import extract, submission
from sklearn.ensemble import RandomForestClassifier
import zipfile 
import pandas as pd

# Load data from CSV file
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv')) # Training data
test_data = pd.read_csv(z.open('test.csv')) # Test data

# Keep only relevant training columns
X_train = train_data['CONTENT'].values # Get just the comment content 
Y_train = train_data['CLASS'].values # Get the labels 

# Keep COMMENT_ID for test data but drop other irrelevant columns
test_ids = test_data['COMMENT_ID'].values  # Save comment IDs for submission
X_test = test_data['CONTENT'].values # Get just the comment content

# Extract feature from Training data
feature_matrix, feature_names = create_feature_matrix(X_train)

# Extract features from Testing data
feature_matrix_test, feature_names_test = create_feature_matrix(X_test)

# Random Forest model 
model = RandomForestClassifier()
model.fit(feature_matrix, Y_train)

# Make predictions on test data
Y_pred = model.predict(feature_matrix_test)

# Create submission dataframe
submission_df = pd.DataFrame({
    'COMMENT_ID': test_ids,
    'CLASS': Y_pred
})
    
# Save submission file
submission_df.to_csv('submissionRF.csv', index=False)

# Accuracy = 91%. Can we improve upon this? 



