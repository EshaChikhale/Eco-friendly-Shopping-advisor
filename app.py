from fastapi import FastAPI, Query, HTTPException
import cv2
import requests
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Load the trained AI model for eco-label prediction
model = joblib.load("C:/Users/varada/Desktop/re-training/eco_label_predictor.pkl")

# Load the dataset to dynamically suggest alternatives
dataset_path = "C:/Users/varada/Desktop/re-training/resaved_dataset.csv"
product_data = pd.read_csv(dataset_path, encoding="utf-8")

def calculate_eco_score_with_ai(features):
    """Use the trained AI model to predict eco-label."""
    input_data = pd.DataFrame([features])
    eco_label = model.predict(input_data)[0]
    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    return eco_label, label_map[eco_label]

def suggest_alternative(product_name, packaging, ingredients):
    """Suggest an alternative product dynamically using text similarity."""
    # Combine packaging and ingredients into a single text field for similarity matching
    product_data["combined_features"] = product_data["Product Name"].fillna("") + " " + \
        product_data["Packaging"].fillna("") + " " + \
        product_data["Ingredients"].fillna("")
    
    target_features = f"{product_name} {packaging} {ingredients}".lower()
    print(f"🔍 Target Features for Matching: {target_features}")

    # Calculate similarity using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(product_data["combined_features"].str.lower())
    target_vector = vectorizer.transform([target_features])
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(target_vector, tfidf_matrix)
    product_data["similarity"] = similarity_scores[0]
    
    print("🔍 Similarity Scores Computed!")

    # Find the most similar eco-friendly product
    eco_friendly_products = product_data[product_data["Eco_Label"] == "High"]
    if eco_friendly_products.empty:
        print("⚠️ No eco-friendly products found!")
        return None
    
    best_match = eco_friendly_products.loc[eco_friendly_products["similarity"].idxmax()]
    print(f"✅ Best Match Found: {best_match['Product Name']}")
    return {
        "name": best_match["Product Name"],
        "brand": best_match["Brand"],
        "barcode": best_match["Barcode"],
        "description": f"Organic hazelnut spread with no palm oil, packed in {best_match['Packaging']}."
    }

def scan_barcode():
    """Open the camera and scan a barcode."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Couldn't read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcode_detector = cv2.barcode_BarcodeDetector()
        retval, decoded_info, _, _ = barcode_detector.detectAndDecodeMulti(gray)

        if retval:
            for barcode_text in decoded_info:
                if barcode_text:
                    print(f"✅ Barcode Detected: {barcode_text}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return barcode_text  # Return the scanned barcode

        cv2.imshow("Barcode Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

@app.get("/")
def home():
    return {"message": "Welcome to the AI-Powered Eco-Friendly Shopping Advisor"}

@app.get("/scan")
def scan_product():
    """Scan a barcode using the camera and fetch product details."""
    barcode = scan_barcode()
    if not barcode:
        return {"error": "❌ No barcode detected. Please try again."}

    return get_product(barcode)

@app.get("/product/{barcode}")
def get_product(
    barcode: str,
    name: str = Query(None, description="Enter product name if not found in database"),
    packaging: str = Query(None, description="Enter packaging details if not found"),
    ingredients: str = Query(None, description="Enter ingredients if not found")
):
    """Fetch product details and return eco-label, or allow user input if product is not found."""
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    response = requests.get(url)

    if response.status_code != 200:
        return {
            "error": "❌ Product not found in OpenFoodFacts.",
            "message": "Please provide the product name, packaging, and ingredients for analysis."
        }

    data = response.json().get("product", {})

    # 🚨 STRICT VALIDATION: Ensure barcode matches
    if not data or data.get("code") != barcode:
        return {
            "error": "❌ Product not found.",
            "message": "You can enter the product name, packaging, and ingredients for analysis."
        }

    # Extract product data
    packaging = data.get("packaging", packaging if packaging else "Not available")
    ingredients = data.get("ingredients_text", ingredients if ingredients else "Not available")

    # Predict eco-label using the AI model
    eco_label, label_text = calculate_eco_score_with_ai({
        "is_plastic": 1 if "plastic" in packaging.lower() else 0,
        "is_glass": 1 if "glass" in packaging.lower() else 0,
        "is_biodegradable": 1 if "biodegradable" in packaging.lower() else 0
    })

    # Suggest alternative dynamically
    alternative = suggest_alternative(data.get("product_name", "Unknown"), packaging, ingredients)

    # Construct the final output
    response_body = {
        "name": data.get("product_name", "Unknown"),
        "brand": data.get("brands", "Unknown"),
        "ingredients": ingredients,
        "packaging": packaging,
        "eco_score": 3 if eco_label == 0 else (6 if eco_label == 1 else 8),
        "eco_explanation": f"{'❌ Poor!' if eco_label == 0 else ('⚠️ Moderate!' if eco_label == 1 else '🌱 Excellent!')} This product has a {'high' if eco_label == 0 else ('moderate' if eco_label == 1 else 'low')} environmental impact.",
        "suggested_alternative": alternative if alternative else "⚠️ No exact match, but consider exploring other sustainable products."
    }
    return response_body