from fastapi import FastAPI, Query
import cv2
import requests

app = FastAPI()

ECO_FRIENDLY_PACKAGING = ["glass", "paper", "cardboard", "biodegradable", "compostable", "recyclable"]
HARMFUL_PACKAGING = ["plastic", "polystyrene", "pet", "non-recyclable"]
ECO_FRIENDLY_INGREDIENTS = ["organic", "natural", "fair trade", "locally sourced"]
HARMFUL_INGREDIENTS = ["palm oil", "artificial additives", "preservatives", "high sugar"]

ECO_ALTERNATIVES = {
    "3017620422003": {
        "name": "Jean Hervé Organic Hazelnut Spread",
        "brand": "Jean Hervé",
        "barcode": "3760011310188",
        "description": "Organic hazelnut spread with no palm oil, packed in a recyclable glass jar."
    },
    "5449000131805": {
        "name": "Lemonaid Organic Lemonade",
        "brand": "Lemonaid",
        "barcode": "4260110210011",
        "description": "A fair-trade organic lemonade with no artificial ingredients, packed in glass bottles."
    }
}

def calculate_eco_score(packaging, ingredients):
    score = 5
    if packaging:
        packaging = packaging.lower()
        for eco_item in ECO_FRIENDLY_PACKAGING:
            if eco_item in packaging:
                score += 2
        for harm_item in HARMFUL_PACKAGING:
            if harm_item in packaging:
                score -= 2
    if ingredients:
        ingredients = ingredients.lower()
        for eco_item in ECO_FRIENDLY_INGREDIENTS:
            if eco_item in ingredients:
                score += 1
        for harm_item in HARMFUL_INGREDIENTS:
            if harm_item in ingredients:
                score -= 1
    return max(1, min(score, 10))

def get_eco_score_explanation(score):
    if score >= 8:
        return "🌱 Excellent! This product is highly eco-friendly."
    elif score >= 6:
        return "✅ Good! This product has a low environmental impact."
    elif score >= 4:
        return "⚠️ Moderate! This product has some environmental concerns."
    else:
        return "❌ Poor! This product has a high environmental impact."

def scan_barcode():
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
                    return barcode_text
        cv2.imshow("Barcode Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return None

@app.get("/")
def home():
    return {"message": "Welcome to the AI-Powered Eco-Friendly Shopping Advisor"}

@app.get("/scan")
def scan_product():
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
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    response = requests.get(url)
    if response.status_code != 200:
        return {
            "error": "❌ Product not found in OpenFoodFacts.",
            "message": "Please provide the product name, packaging, and ingredients for analysis."
        }
    data = response.json().get("product", {})
    if not data or data.get("code") != barcode:
        return {
            "error": "❌ Product not found.",
            "message": "You can enter the product name, packaging, and ingredients for analysis."
        }
    packaging = data.get("packaging", "Not available")
    ingredients = data.get("ingredients_text", "Not available")
    eco_score = calculate_eco_score(packaging, ingredients)
    eco_explanation = get_eco_score_explanation(eco_score)
    alternative = ECO_ALTERNATIVES.get(barcode, "⚠️ No exact match, but consider other sustainable brands.")
    return {
        "name": data.get("product_name", "Unknown"),
        "brand": data.get("brands", "Unknown"),
        "ingredients": ingredients,
        "packaging": packaging,
        "eco_score": eco_score,
        "eco_explanation": eco_explanation,
        "suggested_alternative": alternative
    }
