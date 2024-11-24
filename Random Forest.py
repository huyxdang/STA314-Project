from feature_extractor import extract_text_features, create_feature_matrix
from sklearn.ensemble import RandomForestClassifier
import zipfile 
import pandas as pd
import os

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

# Specify the folder path
folder_path = "Submissions"

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Specify the full file path (folder + file name)
file_path = os.path.join(folder_path, "submissionRF.csv")

# Save the submission file
submission_df.to_csv(file_path, index=False)

# Accuracy = 91%. Can we improve upon this? 



