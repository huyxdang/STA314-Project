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

# Separate spam and non-spam comments
spam_comments = df[df['CLASS'] == 1]
non_spam_comments = df[df['CLASS'] == 0]

# Plot the distributions
plt.figure(figsize=(12, 6))
plt.hist(spam_comments['comment_length'], bins=20, alpha=0.7, label='Spam', density=True)
plt.hist(non_spam_comments['comment_length'], bins=20, alpha=0.7, label='Non-Spam', density=True)
plt.xlabel('Comment Length (by characters)')
plt.ylabel('Density')
plt.title('Distribution of Comment Lengths after removing URL')
plt.legend()
plt.show()

# Calculate summary statistics
spam_stats = spam_comments['comment_length'].describe()
non_spam_stats = non_spam_comments['comment_length'].describe()

# Print the statistics
print("Spam Comment Length Stats:\n", spam_stats)
print("\nNon-Spam Comment Length Stats:\n", non_spam_stats)