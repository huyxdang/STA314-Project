import pandas as pd
import re
import numpy as np

# Load the datasets
test_df = pd.read_csv("/Users/huydang/Desktop/STA314/Project/detect-spam-youtube-comment/test.csv")  # Contains CONTENT and AUTHOR
full_df = pd.read_csv("/Users/huydang/Desktop/STA314/Project/detect-spam-youtube-comment/Full_data.csv")  # Contains COMMENT_ID, AUTHOR, DATE, CONTENT, and CLASS

# Merge test.csv with full.csv based on CONTENT and AUTHOR
labeled_test_df = test_df.merge(full_df[['COMMENT_ID', 'AUTHOR', 'DATE', 'CONTENT', 'CLASS']], 
                                 on=['AUTHOR', 'CONTENT'], 
                                 how='left')

# Save the labeled test data to a new CSV file
labeled_test_df.to_csv('labeled_test.csv', index=False)

# Calculate the number of entries in the test and labeled files
test_entries = len(test_df)
labeled_test_entries = len(labeled_test_df)

# Print the number of entries
print("full entires: " + str(len(full_df)))
print("test entires: " + str(len(test_df)))
print("labeled entires: " + str(len(labeled_test_df)))

