
import cv2
from pyzbar.pyzbar import decode
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def scrape_with_requests(barcode_data):
    print(f"Fetching product details for barcode: {barcode_data} using Requests")

    search_url = f"https://www.google.com/search?q={barcode_data}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all("h3")

        print("\nTop Search Results:")
        for i, result in enumerate(search_results[:5], 1):
            print(f"{i}. {result.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching product details: {e}")
    except AttributeError:
        print("Could not extract product details. Check the site structure.")

def scrape_with_selenium(barcode_data):
    print(f"Fetching product details for barcode: {barcode_data} using Selenium")

    driver = webdriver.Chrome()

    try:
        driver.get(f"https://www.barcodelookup.com/{barcode_data}")
        time.sleep(3)

        product_name = driver.find_element(By.CLASS_NAME, "product-title").text
        product_description = driver.find_element(By.CLASS_NAME, "product-description").text

        print("\nProduct Details:")
        print(f"Name: {product_name}")
        print(f"Description: {product_description}")
    except Exception as e:
        print(f"Error with Selenium scraping: {e}")
    finally:
        driver.quit()

def scrape_product_details(barcode_data):
    print(f"Starting scrape for barcode: {barcode_data}\n")

    scrape_with_requests(barcode_data)
    scrape_with_selenium(barcode_data)

def scan_and_fetch_product_details():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Point your camera at a barcode... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to open the camera.")
            break

        try:
            barcodes = decode(frame)
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type

                if barcode_type == "PDF417":
                    print("PDF417 barcodes are not supported. Skipping...")
                    continue

                print(f"Barcode scanned! Data: {barcode_data}, Type: {barcode_type}")

                x, y, w, h = barcode.rect
                cv2.putText(frame, f"{barcode_data} ({barcode_type})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                scrape_product_details(barcode_data)
        except Exception as e:
            print(f"Error decoding barcode: {e}")

        cv2.imshow("Barcode Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

scan_and_fetch_product_details()
