import pickle
import os

# Define file paths
MODEL_FILE = "eco_score_model.pkl"  # Your trained model file
OUTPUT_FILE = "model_details.txt"   # Output text file

# Check if the model file exists
if not os.path.exists(MODEL_FILE):
    print(f"❌ Error: Model file '{MODEL_FILE}' not found. Ensure the training step was completed.")
    exit()

try:
    # Load the trained model
    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

    # Prepare model details
    model_info = f"✅ Model Loaded: {type(model).__name__}\n\n"
    model_info += "🔍 Model Parameters:\n"
    model_info += "\n".join([f"{key}: {value}" for key, value in model.get_params().items()])

    # Print model details to the console
    print(model_info)

    # Save details to a text file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(model_info)

    print(f"\n📄 Model details saved to '{OUTPUT_FILE}'")

except Exception as e:
    print(f"❌ Error loading the model: {e}")
