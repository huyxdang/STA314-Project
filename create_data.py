import pandas as pd
import re

# Load datasets
test = "/Users/huydang/Desktop/STA314/Project/detect-spam-youtube-comment/test.csv"
full = "/Users/huydang/Desktop/STA314/Project/detect-spam-youtube-comment/Full_data.csv"
output = "/Users/huydang/Desktop/STA314/Project/detect-spam-youtube-comment"

def label_test_data(test_path, full_path, output_path):
    """
    Label test data by matching both AUTHORS and CONTENT with the full dataset.
    
    Parameters:
    test_path (str): Path to test.csv
    full_path (str): Path to full.csv
    output_path (str): Path to save the labeled test data
    """
    # Read the datasets
    test_df = pd.read_csv(test_path)
    full_df = pd.read_csv(full_path)
    
    # Create a unique key combining AUTHOR and CONTENT
    test_df['MATCH_KEY'] = test_df['AUTHOR'] + '||' + test_df['CONTENT']
    full_df['MATCH_KEY'] = full_df['AUTHOR'] + '||' + full_df['CONTENT']
    
    # Create a dictionary mapping the combined key to CLASS
    match_class_map = dict(zip(full_df['MATCH_KEY'], full_df['CLASS']))
    
    # Add CLASS column to test dataset
    test_df['CLASS'] = test_df['MATCH_KEY'].map(match_class_map)
    
    # Remove the temporary MATCH_KEY column
    test_df = test_df.drop('MATCH_KEY', axis=1)
    
    # Save the labeled dataset
    test_df.to_csv(output_path, index=False)
    
    # Print detailed statistics
    total_samples = len(test_df)
    labeled_samples = test_df['CLASS'].notna().sum()
    unlabeled_samples = total_samples - labeled_samples
    
    print("\nLabeling Statistics:")
    print(f"Total samples in test set: {total_samples}")
    print(f"Successfully labeled samples: {labeled_samples}")
    print(f"Unlabeled samples: {unlabeled_samples}")

label_test_data(test,full,output)