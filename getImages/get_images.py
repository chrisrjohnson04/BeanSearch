import os
import json
import re
import requests

API_KEY = "AIzaSyA68QspBJcE7xc6FhPrYEiNnniop7BvVtY"
CSE_ID = "703bd509c025340c2"

def slugify(text: str) -> str:
    """
    Convert a string to a slugified version (lowercase, hyphens instead of spaces,
    remove non-alphanumeric characters except hyphens).
    """
    text = text.lower()
    text = re.sub(r"[^0-9a-z\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text).strip("-")
    return text

coffee_json_path = os.path.join("backend", "coffee.json")

try:
    with open(coffee_json_path, "r", encoding="utf-8") as f:
        beans = json.load(f)
except FileNotFoundError:
    print(f"Error: {coffee_json_path} not found. Ensure you're running this in the project root.")
    beans = []

if not isinstance(beans, list):
    print("Error: coffee.json format is invalid (expected a list of entries).")
    beans = []

for bean in beans:
    bean_name = bean.get("name")
    if not bean_name:
        continue 

    query = f"{bean_name} coffee bean"
    print(f"Searching image for: {query}")

    params = {
        "q": query,
        "cx": CSE_ID,
        "key": API_KEY,
        "searchType": "image",
        "num": 1
    }

    try:
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=5)
    except requests.RequestException as e:
        print(f"HTTP request failed for query '{query}': {e}")
        continue

    if response.status_code != 200:
        print(f"Search API error for '{bean_name}' (status code {response.status_code}).")
        continue

    data = response.json()
    items = data.get("items")
    if not items:
        print(f"No image results found for '{bean_name}'. Skipping.")
        continue

    image_url = items[0].get("link")
    if not image_url:
        print(f"No image URL found for '{bean_name}'. Skipping.")
        continue

    filename = slugify(bean_name) + ".jpg"
    save_path = os.path.join("backend", "static", "coffeeImages", filename)

    try:
        img_response = requests.get(image_url, timeout=10)
        img_response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to download image for '{bean_name}' from URL: {image_url}\n -> {e}")
        continue

    try:
        with open(save_path, "wb") as img_file:
            img_file.write(img_response.content)
        print(f"Saved image for '{bean_name}' as {filename}")
    except Exception as e:
        print(f"Error saving image for '{bean_name}': {e}")
