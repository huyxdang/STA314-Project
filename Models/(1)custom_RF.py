from feature_extractor import extract_text_features, create_feature_matrix
from sklearn.ensemble import RandomForestClassifier
import zipfile 
import pandas as pd
import os

# Load data, separate into training and testing X's and Y's
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv')) # Training data
test_data = pd.read_csv(z.open('test.csv')) # Test data
X_train = train_data['CONTENT'].values # Get x-value of training data
Y_train = train_data['CLASS'].values # Get training data's labels
test_ids = test_data['COMMENT_ID'].values  # Save comment IDs for submission
X_test = test_data['CONTENT'].values # Get just the comment content

# Extract custom features from Training & Testing data 
feature_matrix, feature_names = create_feature_matrix(X_train) # Training data
feature_matrix_test, feature_names_test = create_feature_matrix(X_test) # Testing data

# Random Forest model 
model = RandomForestClassifier()
model.fit(feature_matrix, Y_train)
Y_pred = model.predict(feature_matrix_test) # Make predictions on test data

# Submission
submission_df = pd.DataFrame({
    'COMMENT_ID': test_ids,
    'CLASS': Y_pred
})

folder_path = "Submissions"
if not os.path.exists(folder_path): # Check Submission Folder exists
    os.makedirs(folder_path)

file_path = os.path.join(folder_path, "submissionRF.csv") # Assign Folder Path
submission_df.to_csv(file_path, index=False) # Outputs CSV