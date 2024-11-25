import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import psutil
import zipfile
import os
from tqdm import tqdm

# Suppress warnings
logging.set_verbosity_error()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class CommentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

# BiLSTM Model with RoBERTa embeddings
class RoBERTaBiLSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.2):
        super(RoBERTaBiLSTM, self).__init__()
        
        # Initialize RoBERTa with no pooler
        config = RobertaModel.from_pretrained('roberta-base').config
        config.update({'add_pooling_layer': False})
        self.roberta = RobertaModel.from_pretrained('roberta-base', config=config)
        
        # Freeze RoBERTa parameters
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            roberta_output = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        lstm_output, _ = self.lstm(roberta_output.last_hidden_state)
        final_hidden = self.dropout(lstm_output[:, -1, :])
        output = self.sigmoid(self.fc(final_hidden))
        return output

# Load data
z = zipfile.ZipFile('/Users/huydang/Desktop/STA314/Project/youtube_comments.zip')
train_data = pd.read_csv(z.open('train.csv'))
test_data = pd.read_csv(z.open('test.csv'))

# Extract features and labels
X_train = train_data['CONTENT'].values
Y_train = train_data['CLASS'].values
X_test = test_data['CONTENT'].values
test_ids = test_data['COMMENT_ID'].values

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Training parameters
BATCH_SIZE = 16
MAX_LENGTH = 128
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0

# Create datasets
train_dataset = CommentDataset(X_train, Y_train, tokenizer, MAX_LENGTH)
test_dataset = CommentDataset(X_test, None, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model and optimizer
model = RoBERTaBiLSTM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

print(f"Using device: {device}")

# Training loop
print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].float().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).squeeze()
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

# Generate predictions for test set
print("\nGenerating predictions for test set...")
model.eval()
Y_pred = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask).squeeze()
        predictions = (outputs.cpu().numpy() > 0.5).astype(int)
        Y_pred.extend(predictions)

# Create submission file
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

print(f"\nSubmission file saved to: {file_path}")
print(f"Number of predictions: {len(Y_pred)}")
print(f"Class distribution in predictions: \n{pd.Series(Y_pred).value_counts()}")