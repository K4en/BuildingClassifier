import os
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from duckduckgo_search import DDGS
import time
import random

# Culture queries #
cultures = {
    "egyptian": [
        "ancient egyptian temple", "egyptian pyramid", "egyptian obelisk",
        "egyptian architecture", "egyptian building"
    ],
    "greek": [
        "ancient greek temple", "greek architecture", "greek building",
        "greek column building", "parthenon"
    ],
    "chinese": [
        "traditional chinese building", "chinese temple", "chinese pagoda",
        "ancient chinese architecture", "chinese palace"
    ],
    "indian": [
        "indian temple architecture", "hindu temple", "ancient indian building",
        "mughal architecture", "taj mahal"
    ],
    "european": [
        "european castle", "renaissance architecture", "baroque building",
        "gothic cathedral", "medieval european building"
    ]
}

TARGET_PER_CLASS = 500
SAVE_DIR = "Data/Raw"
MAX_PER_QUERY = 150

# Helpers #
def sanitize_filename(text):
    return "".join(c if c.isalnum() or c in ("-","_") else "_" for c in text)

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=5)
        content_type = response.headers.get("content-type")
        if not content_type or not content_type.startswith("image"):
            raise ValueError("URL did not return image")

        image = Image.open(BytesIO(response.content))

        if image.mode not in ("RGB", "RGBA", "L"):
            raise ValueError(f"Unsupported image mode: {image.mode}")

        rgb_image = image.convert("RGB")
        rgb_image.save(save_path, format="JPEG")
        return True
    except (UnidentifiedImageError, ValueError, Exception) as e:
        print(f"Error downloading image: {e}")
        return False

# Main Loop #
def download_category(category, query_list, target_count):
    os.makedirs(os.path.join(SAVE_DIR, category), exist_ok=True)
    downloaded = 0
    failed_log = []
    ddgs = DDGS()

    for query in query_list:
        if downloaded >= target_count:
            break # Stop querying if target already met
        print(f"Searching for: {query}")
        try:
            results = ddgs.images(query, max_results=MAX_PER_QUERY)
        except Exception as e:
            print(f"Query failed: {query}-{e}")
            continue

        for result in results:
            if downloaded >= target_count:
                print(f"Reached target for {category}: {result}")
                break

            image_url = result.get("image")
            if not image_url:
                continue

            filename = f"{category}_{downloaded}.jpg"
            save_path = os.path.join(SAVE_DIR, category, filename)
            if download_image(image_url, save_path):
                downloaded += 1
                time.sleep(1.3)


    print(f"Finished: '{category}' with {downloaded} images")
    if failed_log:
        with open(f"failed_{category}.txt", "w") as f:
            for url in failed_log:
                f.write(url + "\n")

# Run #
if __name__ == "__main__":
    for culture, queries in cultures.items():
        download_category(culture, queries, TARGET_PER_CLASS)

    print("Download completed. Banzai!! Yippe!")


