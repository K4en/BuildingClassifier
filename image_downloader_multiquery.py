import os
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from duckduckgo_search import DDGS
import time
import random

# Config #
queries = {
    "Residential": [
        "residential building", "modern house", "suburban home",
        "apartment block", "urban home", "family house"
    ],
    "Industrial": [
        "industrial building", "factory exterior", "warehouse",
        "manufacturing plant", "power station", "industrial architecture"
    ]
}

TARGET_PER_CLASS = 500
SAVE_DIR = "Data/Raw"
MAX_PER_QUERY = 150 # stop early if enough valid image

# Helpers #
def sanitize_filename(text):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=5)
        content_type = response.headers.get("content-type")
        if not content_type or not content_type.startswith("image"):
            raise ValueError("URL did not return an image")

        image = Image.open(BytesIO(response.content))

        if image.mode not in ("RGB", "RGBA", "L"):
            raise ValueError(f"Not supported image format: {image.mode}")

        rgb_image = image.convert("RGB")
        rgb_image.save(save_path, format="JPEG")
        return True
    except (UnidentifiedImageError, ValueError, Exception) as e:
        print(f"Error downloading image: {e}")
        return False

# Main download loop #
def download_category(category, query_list, target_count):
    os.makedirs(os.path.join(SAVE_DIR, category), exist_ok=True)
    downloaded = 0
    failed_log = []
    used_urls = set()

    with DDGS() as ddgs:
        for query in query_list:
            print(f"\n Searching: {query}")
            for result in ddgs.images(query, max_results=MAX_PER_QUERY):
                url = result.get("image")
                if not url or url in used_urls:
                    continue

                used_urls.add(url)
                filename = f"{category}_{downloaded}.jpg"
                save_path = os.path.join(SAVE_DIR, category, filename)

                if download_image(url, save_path):
                    downloaded += 1
                    print(f" [{downloaded}/{target_count}] {url}")
                else:
                    failed_log.append(url)

                if downloaded >= target_count:
                    print(f" Reached target {target_count} images for '{category}'")
                    return

            time.sleep(random.uniform(1, 2)) # polite pause

    print(f" Finished '{category}' with {downloaded} images. Some may have possibly for sure failed...")
    if failed_log:
        with open(f"failed_{category}.txt", "w") as f:
            for url in failed_log:
                f.write(url + "\n")

# Run all this #
if __name__ == "__main__":
    for category, qlist in queries.items():
        download_category(category, qlist, TARGET_PER_CLASS)

    print("Download complete! Yay! Banzai! Cheers!")