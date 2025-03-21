import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

# Define file paths
MODEL_FILE = "eco_score_model.pkl"  # Trained model
TRAINED_DATA_FILE = "train_data.csv"  # Training dataset
TEST_DATA_FILE = "test_data.csv"  # Test dataset
TEST_OUTPUT_FILE = "test_results.csv"  # Output file

# ✅ Check if the model file exists
if not os.path.exists(MODEL_FILE):
    print(f"❌ Error: Model file '{MODEL_FILE}' not found.")
    exit()

# ✅ Load the trained model
try:
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ✅ Load the training dataset to extract feature names & train OneHotEncoder
if not os.path.exists(TRAINED_DATA_FILE):
    print(f"❌ Error: Training dataset '{TRAINED_DATA_FILE}' not found.")
    exit()

try:
    train_data = pd.read_csv(TRAINED_DATA_FILE)

    # ✅ Drop non-feature columns before encoding
    train_data = train_data.drop(columns=["name", "brand", "eco_score"], errors="ignore")

    # ✅ Handle One-Hot Encoding for `ingredients`
    if "ingredients" in train_data.columns:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_ingredients = encoder.fit_transform(train_data[["ingredients"]])
        encoded_ingredients_df = pd.DataFrame(encoded_ingredients, columns=encoder.get_feature_names_out(["ingredients"]))
        train_data = pd.concat([train_data, encoded_ingredients_df], axis=1)

    # Drop original `ingredients` column
    train_data = train_data.drop(columns=["ingredients"], errors="ignore")

    # ✅ Extract feature names from training data
    feature_names = train_data.columns.tolist()
    print(f"✅ Feature names extracted from {TRAINED_DATA_FILE}")

except Exception as e:
    print(f"❌ Error processing training dataset: {e}")
    exit()

# ✅ Load the test dataset
if not os.path.exists(TEST_DATA_FILE):
    print(f"❌ Error: Test dataset '{TEST_DATA_FILE}' not found.")
    exit()

try:
    test_data = pd.read_csv(TEST_DATA_FILE)

    # ✅ Drop irrelevant columns before encoding
    test_data = test_data.drop(columns=["name", "brand"], errors="ignore")

    # ✅ Apply One-Hot Encoding for `ingredients` using the same encoder from training
    if "ingredients" in test_data.columns:
        encoded_ingredients = encoder.transform(test_data[["ingredients"]])
        encoded_ingredients_df = pd.DataFrame(encoded_ingredients, columns=encoder.get_feature_names_out(["ingredients"]))
        test_data = pd.concat([test_data, encoded_ingredients_df], axis=1)

    # Drop original `ingredients` column
    test_data = test_data.drop(columns=["ingredients"], errors="ignore")

    # ✅ Ensure test data matches training feature structure
    for column in feature_names:
        if column not in test_data.columns:
            test_data[column] = 0  # Add missing features with default value 0

    # Drop any extra columns that do not match training features
    test_data = test_data[feature_names]

    print(f"✅ Test data processed successfully from {TEST_DATA_FILE}")

except Exception as e:
    print(f"❌ Error processing test dataset: {e}")
    exit()

# ✅ Make predictions
try:
    predictions = model.predict(test_data)

    # Convert predictions to single-digit format (0-9)
    predictions = np.clip(np.round(predictions).astype(int), 0, 9)

    # Store predictions
    test_data["Predicted Eco Score"] = predictions

    # Save predictions to CSV file
    test_data.to_csv(TEST_OUTPUT_FILE, index=False)

    print("\n🔍 Test Results:")
    print(test_data[["Predicted Eco Score"]].head())  # Display first few rows

    print(f"\n📄 Test results saved to '{TEST_OUTPUT_FILE}'")

except Exception as e:
    print(f"❌ Error making predictions: {e}")
