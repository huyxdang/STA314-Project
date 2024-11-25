import pandas as pd
import numpy as np
import fasttext.util
import fasttext
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import psutil
import zipfile
import os
from tqdm import tqdm

# Download FastText model if not already downloaded
print("Loading FastText model...")
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

def get_text_embedding(text, ft_model):
    """Get FastText embedding for a text string"""
    # Clean and tokenize text
    text = str(text).lower()
    # Get embedding
    return ft_model.get_sentence_vector(text)

def create_embeddings_matrix(texts, ft_model, batch_size=1000):
    """Create embeddings matrix for a list of texts"""
    n_texts = len(texts)
    embeddings = np.zeros((n_texts, 300))  # FastText embeddings are 300-dimensional
    
    for i in tqdm(range(0, n_texts, batch_size), desc="Creating embeddings"):
        batch_texts = texts[i:min(i + batch_size, n_texts)]
        embeddings[i:i + len(batch_texts)] = np.array([
            get_text_embedding(text, ft_model) for text in batch_texts
        ])
    
    return embeddings

# Memory usage function
def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

print("Loading data...")
# Load data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))
test_data = pd.read_csv(z.open('test.csv'))

# Extract features and labels
X_train = train_data['CONTENT'].values
Y_train = train_data['CLASS'].values
X_test = test_data['CONTENT'].values
test_ids = test_data['COMMENT_ID'].values

print("Creating FastText embeddings for training data...")
start_time = time.time()
X_train_embeddings = create_embeddings_matrix(X_train, ft)
train_embedding_time = time.time() - start_time
print(f"Training embeddings created in {train_embedding_time:.2f} seconds")

print("Creating FastText embeddings for test data...")
start_time = time.time()
X_test_embeddings = create_embeddings_matrix(X_test, ft)
test_embedding_time = time.time() - start_time
print(f"Test embeddings created in {test_embedding_time:.2f} seconds")

# Free up memory
del ft
import gc
gc.collect()

# XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',  # For faster training
    'random_state': 42
}

print("Training XGBoost model...")
# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_embeddings, label=Y_train)
dtest = xgb.DMatrix(X_test_embeddings)

# Train model
start_train_time = time.time()
model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train')],
    verbose_eval=10
)
train_time = time.time() - start_train_time
print(f"\nModel trained in {train_time:.2f} seconds")

# Make predictions
print("\nGenerating predictions...")
start_inference_time = time.time()
Y_pred_proba = model.predict(dtest)
Y_pred = (Y_pred_proba > 0.5).astype(int)
inference_time = time.time() - start_inference_time
print(f"Predictions generated in {inference_time:.2f} seconds")

# Create submission file
print("\nCreating submission file...")
submission_df = pd.DataFrame({
    'COMMENT_ID': test_ids,
    'CLASS': Y_pred
})

# Create Submissions folder if it doesn't exist
folder_path = "Submissions"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save submission file
file_path = os.path.join(folder_path, "submission.csv")
submission_df.to_csv(file_path, index=False)

# Print summary statistics
print(f"\nSubmission file saved to: {file_path}")
print(f"Number of predictions: {len(Y_pred)}")
print(f"Class distribution in predictions: \n{pd.Series(Y_pred).value_counts()}")

# Print performance metrics
print("\nPerformance Metrics:")
print(f"Training Time: {train_time:.2f} seconds")
print(f"Inference Time: {inference_time:.2f} seconds")
print(f"Total Embedding Time: {train_embedding_time + test_embedding_time:.2f} seconds")
print(f"Peak RAM Usage: {memory_usage():.2f} MB")

# Feature importance analysis
importance_scores = model.get_score(importance_type='gain')
importance_df = pd.DataFrame(
    list(importance_scores.items()),
    columns=['Feature', 'Importance']
).sort_values('Importance', ascending=False)

# Save feature importance
importance_file_path = os.path.join(folder_path, "feature_importance.csv")
importance_df.to_csv(importance_file_path, index=False)
print(f"Feature importance saved to: {importance_file_path}")