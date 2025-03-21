import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder

# Download stopwords (only needed once)
nltk.download("stopwords")

# Load dataset
DATA_FILE = "cleaned_product_data.csv"

# Debug: Print current working directory
current_dir = os.getcwd()
print(f"📂 Current working directory: {current_dir}")

# Check if the cleaned dataset exists
if not os.path.exists(DATA_FILE):
    print(f"❌ Error: '{DATA_FILE}' not found in {current_dir}.")
    print("🔍 Please ensure the preprocessing script has run successfully and saved 'cleaned_product_data.csv' in the correct location.")
    exit()

df = pd.read_csv(DATA_FILE)

# Handle missing values: Replace "Unknown" and "available" with NaN
df.replace(["Unknown", "available"], pd.NA, inplace=True)

# Drop rows where key columns contain NaN
required_columns = ["name", "brand", "packaging", "ingredients", "eco_score"]
df.dropna(subset=required_columns, inplace=True)

# Remove "available" from text columns
df = df[~df["ingredients"].str.contains("available", na=False, case=False)]
df = df[~df["packaging"].str.contains("available", na=False, case=False)]

# Convert text to lowercase and remove stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if pd.isna(text):  # Ensure text isn't NaN
        return ""
    words = text.lower().split()  # Convert to lowercase & split
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)  # Join words back

# Apply text cleaning
text_columns = ["name", "brand", "ingredients"]
for col in text_columns:
    df[col] = df[col].apply(clean_text)

# One-Hot Encoding for 'packaging' column
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_packaging = encoder.fit_transform(df[["packaging"]])

# Convert encoded columns to DataFrame
packaging_columns = [f"packaging_{cat}" for cat in encoder.categories_[0]]
packaging_df = pd.DataFrame(encoded_packaging, columns=packaging_columns)

# Merge and drop original 'packaging' column
df = pd.concat([df, packaging_df], axis=1).drop(columns=["packaging"])

# Save the encoded data
ENCODED_CSV = "encoded_data.csv"
df.to_csv(ENCODED_CSV, index=False)

print(f"✅ Feature engineering complete! Encoded file saved as {ENCODED_CSV}")
