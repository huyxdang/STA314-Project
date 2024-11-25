'''
import pandas as pd
import matplotlib.pyplot as plt
import re
import mplcursors

# Load your dataset
df = pd.read_csv('/Users/huydang/Desktop/STA314-Project/all.csv')


def count_exclamation_question(comment):
    return len(re.findall(r'[!?]', comment))

df['!?_count'] = df['CONTENT'].apply(count_exclamation_question)

# Separate spam and non-spam comments
spam_comments = df[df['CLASS'] == 1]  # Assuming '1' indicates spam
non_spam_comments = df[df['CLASS'] == 0]  # Assuming '0' indicates non-spam

# Plot the distribution for spam and non-spam


# Plot the histograms
plt.figure(figsize=(12, 6))
bars_spam = plt.hist(spam_comments['!?_count'], bins=5, alpha=0.7, label='Spam', density=True, color='red')
bars_non_spam = plt.hist(non_spam_comments['!?_count'], bins=5, alpha=0.7, label='Non-Spam', density=True, color='blue')

plt.xlim(0,200)
plt.xlabel('Number of "!" and "?" Punctuations')
plt.ylabel('Density')
plt.title('Distribution of "!" and "?" in Spam vs. Non-Spam Comments')
plt.legend()

# Add an interactive cursor
cursor = mplcursors.cursor(highlight=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(
    f"x: {sel.target[0]:.2f}\ny: {sel.target[1]:.4f}"
))

plt.show()
# Intersection point = 38 
'''