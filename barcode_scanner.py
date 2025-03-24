import cv2
import requests
from pyzbar.pyzbar import decode
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ✅ Load the fine-tuned model
model_path = "bert_base_food_classifier"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# ✅ Function to fetch product details from Open Food Facts API
def get_product_name(barcode):
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "product" in data and "product_name" in data["product"]:
            return data["product"]["product_name"]
    return None

# ✅ Function to classify product category
def classify_product(product_name):
    inputs = tokenizer(product_name, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

# ✅ Barcode scanner using OpenCV & pyzbar
def scan_barcode():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        _, frame = cap.read()
        barcodes = decode(frame)

        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")  # Convert bytes to string
            print(f"✅ Scanned Barcode: {barcode_data}")
            product_name = get_product_name(barcode_data)
            if product_name:
                print(f"📦 Product Found: {product_name}")
                category = classify_product(product_name)
                print(f"🏷️ Predicted Category: {category}")
            else:
                print("❌ Product not found.")
            cap.release()
            cv2.destroyAllWindows()
            return

        cv2.imshow("Barcode Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ Start barcode scanner
scan_barcode()
