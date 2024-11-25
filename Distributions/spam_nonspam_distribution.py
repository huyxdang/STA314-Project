from Functions.data_functions import extract
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data, test_data = extract()

train_labels = train_data['CLASS'].values # Retrieve labels of training data

# Count the number of spam (1) and non-spam (0) comments
unique, counts = np.unique(train_labels, return_counts=True)
label_distribution = dict(zip(unique, counts))

# Plot the distribution of spam vs. non-spam comments
plt.figure(figsize=(8,6))
plt.bar(['Non-Spam (0)', 'Spam(1)'], counts, color=['blue', 'red'], edgecolor='black')
plt.title("Distribution of Spam vs. Non-Spam in Training data", fontsize=16)
plt.xlabel("Class", fontsize=12)
plt.ylabel('Number of Commetns', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()