import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

# File paths
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"

# Load training and testing datasets
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

# Ensure 'eco_score' exists in the dataset
if "eco_score" not in df_train.columns or "eco_score" not in df_test.columns:
    print("❌ Error: 'eco_score' column missing in dataset.")
    exit()

# Separate features (X) and target variable (y)
X_train = df_train.drop(columns=["eco_score"])
y_train = df_train["eco_score"]
X_test = df_test.drop(columns=["eco_score"])
y_test = df_test["eco_score"]

# 🔹 One-Hot Encode `ingredients` Column (if exists)
if "ingredients" in X_train.columns:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    # Fit on train, transform both train & test
    X_train_encoded = encoder.fit_transform(X_train[["ingredients"]])
    X_test_encoded = encoder.transform(X_test[["ingredients"]])
    
    # Convert to DataFrame
    encoded_columns = encoder.get_feature_names_out(["ingredients"])
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
    
    # Merge encoded data & drop original "ingredients" column
    X_train = pd.concat([X_train.drop(columns=["ingredients"]), X_train_encoded], axis=1)
    X_test = pd.concat([X_test.drop(columns=["ingredients"]), X_test_encoded], axis=1)

# 🔹 Drop Non-Numeric Columns
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Train the model
print("🚀 Training the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Model trained successfully! Mean Absolute Error: {mae:.2f}")

# Save the trained model
MODEL_FILE = "eco_score_model.pkl"
with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)

print(f"📁 Model saved as '{MODEL_FILE}'")
