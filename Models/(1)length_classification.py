import pandas as pd
import matplotlib.pyplot as plt
import re

# Function to replace URLs
def replace_urls_with_placeholder(comment):
    return re.sub(r'http[s]?://\S+|www\.\S+', 'URL', comment)

# Load your dataset
df = pd.read_csv('/Users/huydang/Desktop/STA314-Project/all.csv')

# Replace URLs in comments
df['CONTENT'] = df['CONTENT'].apply(replace_urls_with_placeholder)

# Calculate comment lengths
df['comment_length'] = df['CONTENT'].apply(len)

# Add "length classification" column
df['length classification'] = df['comment_length'].apply(lambda x: 1 if x >= 200 else 2) # Classify comment as 2 if length exceeds 200

# Verify the new column
print(df[['CONTENT', 'comment_length', 'length classification']].head())

