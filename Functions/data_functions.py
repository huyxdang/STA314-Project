import pandas as pd
import zipfile 

def extract(file_path):
    """
    Load and extract zipfile into training and testing data sets
    
    Parameter: 
    file_path(str): The file path of the zip file
    
    Returns: 
    train_data: Training data
    test_data: Testing data
    """
    # Load data from csv file 
    z = zipfile.ZipFile(file_path)
    train_data = pd.read_csv(z.open("train.csv")) # Training data
    test_data = pd.read_csv(z.open("test.csv"))
    return train_data, test_data

def submission(ids, y_pred):
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'COMMENT_ID': test_comment_ids,
        'CLASS': Y_pred
    })
    
    # Save submission file
    submission_df.to_csv('submission1.csv', index=False)

