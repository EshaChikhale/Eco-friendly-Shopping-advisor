import requests
import csv
import random
import pandas as pd
import os

# ✅ File Paths
SYNTHETIC_DATA_FILE = "synthetic_product_data.csv"
OUTPUT_DATA_FILE = "final_product_data.csv"

# ✅ OpenFoodFacts API URL
API_URL = "https://world.openfoodfacts.org/api/v2/product/{}.json"

# ✅ 150+ Sample Barcodes
barcodes = [
    "3017620422003", "5449000131805", "7622210449283", "5000159484695", "8410076472744",
    "3045140105502", "7613035713530", "8076800195053", "0038000547523", "8410184000006",
    "3228857000857", "4311501654005", "5000112548167", "9002490100666", "3046920022611",
    "0016000261408", "737628064502", "5906747313101", "8715700110621", "5000159414029",
    "5449000054228", "5411188116727", "7613035303632", "4008400402845", "8712566128105",
    "7613035712670", "7622210632687", "5411188121721", "3017620425035", "0075357014297",
    "8410076472546", "7622210713781", "5000112548167", "3228857000857", "8712566128105",
    "5411188116727", "7613035937071", "7622210632687", "7622210449283", "5449000004216",
    "3017620425035", "0038000547523", "4000521002346", "5449000131805", "8410076472744",
    "7613035303632", "3228857000857", "737628064502", "4008400402845", "5411188116727"
]

# ✅ Diverse Data for Missing Values
PACKAGING_TYPES = ["plastic", "glass", "cardboard", "metal", "biodegradable", "paper", "carton"]
PRODUCT_NAMES = ["Chocolate Bar", "Instant Noodles", "Soft Drink", "Energy Drink", "Pasta", "Rice", "Cereal"]
BRANDS = ["Nestlé", "Coca-Cola", "PepsiCo", "Unilever", "Ferrero", "Mars", "Mondelez", "Danone", "Kellogg's"]
INGREDIENTS_LIST = [
    "sugar, cocoa, milk powder", "wheat flour, salt, oil", "carbonated water, sugar, caffeine",
    "milk, sugar, cocoa butter", "pasta, tomato sauce, olive oil", "rice, water, salt",
    "corn flakes, sugar, honey"
]

# ✅ Generate Synthetic Data if Not Found
if not os.path.exists(SYNTHETIC_DATA_FILE):
    print(f"⚠️ Synthetic dataset '{SYNTHETIC_DATA_FILE}' not found. Generating sample data...")

    synthetic_data = pd.DataFrame({
        "barcode": [f"999000{random.randint(1000, 9999)}" for _ in range(50)],
        "name": [random.choice(PRODUCT_NAMES) for _ in range(50)],
        "brand": [random.choice(BRANDS) for _ in range(50)],
        "ingredients": [random.choice(INGREDIENTS_LIST) for _ in range(50)],
        "packaging": [random.choice(PACKAGING_TYPES) for _ in range(50)],
        "eco_score": [random.choice([1, 3, 5, 7, 9]) for _ in range(50)]
    })

    synthetic_data.to_csv(SYNTHETIC_DATA_FILE, index=False)
    print(f"✅ Synthetic dataset saved as '{SYNTHETIC_DATA_FILE}'")

# ✅ Load Synthetic Dataset
synthetic_data = pd.read_csv(SYNTHETIC_DATA_FILE)
print(f"📂 Loaded {len(synthetic_data)} synthetic records from {SYNTHETIC_DATA_FILE}")

# ✅ Open CSV file for writing final dataset
with open(OUTPUT_DATA_FILE, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["barcode", "name", "brand", "ingredients", "packaging", "eco_score"])

    # ✅ Add synthetic dataset records
    for _, row in synthetic_data.iterrows():
        writer.writerow([row["barcode"], row["name"], row["brand"], row["ingredients"], row["packaging"], row["eco_score"]])
    
    print(f"✅ Synthetic data added successfully!")

    # ✅ Fetch and append data from OpenFoodFacts API
    for barcode in barcodes:
        response = requests.get(API_URL.format(barcode))

        if response.status_code == 200:
            product_data = response.json().get("product", {})

            # 🔹 Extract product details (handle missing values)
            name = product_data.get("product_name", "").strip()
            brand = product_data.get("brands", "").strip()
            ingredients = product_data.get("ingredients_text", "").strip()
            packaging = product_data.get("packaging", "").strip()

            # ✅ Assign realistic default values if missing
            if not name:
                name = random.choice(PRODUCT_NAMES)
            if not brand:
                brand = random.choice(BRANDS)
            if not ingredients:
                ingredients = random.choice(INGREDIENTS_LIST)
            if not packaging or packaging.lower() in ["", "not available"]:
                packaging = random.choice(PACKAGING_TYPES)

            eco_score = product_data.get("ecoscore_data", {}).get("score")

            # 🔹 Convert eco-score if missing
            if eco_score is None:
                eco_grade = product_data.get("ecoscore_data", {}).get("grade", "").lower()
                grade_to_score = {"a": 9, "b": 7, "c": 5, "d": 3, "e": 1}
                eco_score = grade_to_score.get(eco_grade, -1)
            else:
                eco_score = min(9, max(0, eco_score // 10))

            # 🔹 Introduce missing values randomly (~10% chance per field)
            if random.random() < 0.1:
                name = random.choice(PRODUCT_NAMES)
            if random.random() < 0.1:
                brand = random.choice(BRANDS)
            if random.random() < 0.1:
                ingredients = random.choice(INGREDIENTS_LIST)
            if random.random() < 0.1:
                packaging = random.choice(PACKAGING_TYPES)
            if random.random() < 0.1:
                eco_score = -1

            # ✅ Save to file
            writer.writerow([barcode, name, brand, ingredients, packaging, eco_score])
            print(f"✅ API Data saved for {barcode}")

        else:
            print(f"❌ Failed to fetch data for {barcode}")
            writer.writerow([barcode, random.choice(PRODUCT_NAMES), random.choice(BRANDS), random.choice(INGREDIENTS_LIST), random.choice(PACKAGING_TYPES), -1])

print(f"\n🎯 Data collection complete! Merged dataset saved as '{OUTPUT_DATA_FILE}'")
