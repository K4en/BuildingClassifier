import os
import requests
import shutil
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO

def download_images(query, folder, max_images=100 ):
    os.makedirs(folder, exist_ok=True)
    count = 0

    with DDGS() as ddgs:
        for result in ddgs.images(query, max_results=max_images):
            url = result.get("image")
            if not url:
                continue

            try:
                response = requests.get(url, timeout=5)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(os.path.join(folder, f"{query}_{count}.jpg"))
                count += 1
            except Exception as e:
                print(f"Failed to download image {url}: {e}")

            if count >= max_images:
                break

# Create folders and download images
download_images("residential building", "Data/Raw/Residential", max_images=600)
download_images("industrial building", "Data/Raw/Industrial", max_images=600)