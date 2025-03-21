import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# ✅ Ensure required NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ✅ Load dataset
CSV_FILE = "final_product_data.csv"  # Ensure this matches your cleaned dataset
df = pd.read_csv(CSV_FILE)

# ✅ Define values to treat as missing
MISSING_VALUES = ["Unknown", "Product A", "Brand A", "Not Available", "available"]

# ✅ Replace missing values with NaN
df.replace(MISSING_VALUES, pd.NA, inplace=True)

# ✅ Drop rows where any key column is NaN
required_columns = ["name", "brand", "packaging", "ingredients", "eco_score"]
df.dropna(subset=required_columns, inplace=True)

# ✅ Remove invalid values in "ingredients" & "packaging"
df = df[~df["ingredients"].str.contains("not available", na=False, case=False)]
df = df[~df["packaging"].str.contains("not available", na=False, case=False)]

# ✅ Define stopwords & punctuation
stop_words = set(stopwords.words("english"))
punctuation_table = str.maketrans("", "", string.punctuation)

def clean_text(text):
    """Clean text: lowercase, remove punctuation, remove stopwords."""
    if pd.isna(text):  # Ensure text isn't NaN
        return ""
    
    text = text.lower().translate(punctuation_table)  # Lowercase & remove punctuation
    words = text.split()  # Basic tokenization to avoid `punkt_tab` issue
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    
    return " ".join(filtered_words)  # Join words back

# ✅ Apply text cleaning correctly (Fixed applymap issue)
text_columns = ["name", "brand", "packaging", "ingredients"]
for col in text_columns:
    df[col] = df[col].astype(str).apply(clean_text)

# ✅ Save cleaned dataset
CLEANED_CSV = "cleaned_product_data.csv"
df.to_csv(CLEANED_CSV, index=False)

print(f"🎯 Data cleaning complete! Cleaned file saved as '{CLEANED_CSV}'")
