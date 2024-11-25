import pandas as pd
import numpy as np
import time
import psutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import fasttext.util
import xgboost as xgb
import zipfile

# Load data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))  # Training data
test_data = pd.read_csv(z.open('test.csv'))  # Test data

# Extract features and labels
X_train = train_data['CONTENT'].values  # Text content for training
Y_train = train_data['CLASS'].values  # Labels for training
X_test = test_data['CONTENT'].values  # Text content for testing
test_ids = test_data['COMMENT_ID'].values  # Comment IDs for the test data

# Load FastText pre-trained embeddings
fasttext.util.download_model('en', if_exists='ignore')  # Download English FastText embeddings
ft = fasttext.load_model('cc.en.300.bin')  # Load 300-dimensional FastText embeddings

# Function to convert sentences to FastText embeddings
def sentence_to_vector(sentence, model, embedding_dim=300):
    words = sentence.split()
    vectors = [model.get_word_vector(word) for word in words]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)  # Average word embeddings
    else:
        return np.zeros(embedding_dim)  # Default zero vector for empty/unknown words

# Generate FastText embeddings for the training and test datasets
X_train_embeddings = np.array([sentence_to_vector(text, ft) for text in X_train])
X_test_embeddings = np.array([sentence_to_vector(text, ft) for text in X_test])

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_metrics = []

def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

# Training and validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_embeddings, Y_train)):
    print(f"Fold {fold + 1}")
    X_train_fold, X_val_fold = X_train_embeddings[train_idx], X_train_embeddings[val_idx]
    Y_train_fold, Y_val_fold = Y_train[train_idx], Y_train[val_idx]

    # Measure training time
    start_train_time = time.time()
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train_fold, Y_train_fold)
    train_time = time.time() - start_train_time

    # Measure inference time
    start_inference_time = time.time()
    Y_val_pred = model.predict(X_val_fold)
    inference_time = time.time() - start_inference_time

    # Calculate metrics
    accuracy = accuracy_score(Y_val_fold, Y_val_pred)
    precision = precision_score(Y_val_fold, Y_val_pred, average='weighted')
    recall = recall_score(Y_val_fold, Y_val_pred, average='weighted')
    f1 = f1_score(Y_val_fold, Y_val_pred, average='weighted')
    
    # AUC-ROC (requires probability predictions)
    Y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
    auc_roc = roc_auc_score(Y_val_fold, Y_val_pred_proba)

    # Record metrics
    fold_metrics.append({
        "Fold": fold + 1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc,
        "Training Time (s)": train_time,
        "Inference Time (s)": inference_time,
        "RAM Usage (MB)": memory_usage()
    })

# Create a DataFrame to display metrics
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df)

# Average metrics across folds
average_metrics = metrics_df.drop(columns=["Fold"]).mean(axis=0).to_dict()
print("\nAverage Metrics Across Folds:")
for metric, value in average_metrics.items():
    print(f"{metric}: {value:.4f}")

# Train the final model on the entire training dataset and make predictions on the test dataset
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
final_model.fit(X_train_embeddings, Y_train)
Y_test_pred = final_model.predict(X_test_embeddings)

# Create submission file
submission_df = pd.DataFrame({
    'COMMENT_ID': test_ids,
    'CLASS': Y_test_pred
})
submission_df.to_csv('submissionXGB_FastText.csv', index=False)
