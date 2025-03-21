import pandas as pd
import os
from sklearn.model_selection import train_test_split

# File path
DATA_FILE = "encoded_data.csv"

# Debug: Print current directory
current_dir = os.getcwd()
print(f"\U0001F4C2 Current working directory: {current_dir}")

# Check if the file exists
if not os.path.exists(DATA_FILE):
    print(f"\u274C Error: '{DATA_FILE}' not found in {current_dir}.")
    print("\U0001F50D Please ensure the encoding script has run successfully and saved 'encoded_data.csv' in the correct location.")
    exit()

# Load the encoded dataset
df = pd.read_csv(DATA_FILE)

# Split dataset: 80% training, 20% testing
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save train and test datasets
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("\u2705 Train-test split complete! Files saved:")
print("   - train_data.csv")
print("   - test_data.csv")
