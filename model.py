# Import packages 
import pandas as pd

# Load data
file_path = '/Users/huydang/Desktop/STA314 (ML)/Project/detect-spam-youtube-comment/train.csv'
data = pd.read_csv(file_path)

# Print the fiew few rows to confirm it's loaded correctly 
print(data.head())