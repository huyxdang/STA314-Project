from feature_extractor import create_feature_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the aggregated training data
train_data = pd.read_csv('/Users/huydang/Desktop/STA314-Project/all.csv')

# Extract relevant columns
X = train_data['CONTENT'].values  # Get the comment content
y = train_data['CLASS'].values  # Get the labels

# Extract feature matrix from the training data
feature_matrix, feature_names = create_feature_matrix(X)

# Initialize the model
model = RandomForestClassifier()

# Initialize Stratified K-Fold with 10 folds
n_splits = 10  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Perform Stratified K-Fold Cross-Validation
fold = 1
accuracies = []

for train_index, val_index in skf.split(feature_matrix, y):
    print(f"Fold {fold}:")
    
    # Split data into training and validation sets
    X_train, X_val = feature_matrix[train_index], feature_matrix[val_index]
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

