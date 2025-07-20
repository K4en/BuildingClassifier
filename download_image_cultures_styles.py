import os
import shutil
from duckduckgo_search import DDGS
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import time
import random

# Constants #
DATA_DIR = Path("Data/Phase 3/Raw")
CULTURE_STYLES = {
    "egyptian": ["ancient egyptian temple", "egyptian pyramid architecture", "egyptian obelisk design", "ancient egyptian architecture", "ancient egyptian relief walls"],
    #"greek": ["ancient greek temple", "greek amphitheater", "greek column doric ionic corinthian"],
    #"chinese": ["ancient chinese palace", "traditional chinese pagoda", "chinese courtyard house"],
    #"indian": ["ancient indian temple", "mughal architecture", "south indian temple gopuram"],
    #"european": ["gothic cathedral architecture", "renaissance architecture", "baroque palace design"]
}
TARGET_PER_STYLE = 100
MAX_PER_QUERY = 50
DELAY_RANGE = (1.0, 2.5)

def download_images(style_dir: Path, queries: list[str], target_count: int):
    downloaded = 0
    seen_urls = set()
    style_dir.mkdir(parents=True, exist_ok=True)
    with DDGS() as ddgs:
        for query in queries:
            if downloaded >= target_count:
                break

            print(f"Searching for: {query}")
            results = ddgs.images(query, max_results=MAX_PER_QUERY)
            if not results:
                print(f"No results for {query}")
                continue

            for result in results:
                if downloaded >= target_count:
                    break

                url = result.get("image")
                if not url or url in seen_urls:
                    continue

                try:
                    response = requests.get(url, timeout=10)
                    image = Image.open(BytesIO(response.content)).convert("RGB")

                    filename = f"{query}-{downloaded}.jpg"
                    save_path = style_dir / filename
                    image.save(save_path)
                    downloaded += 1
                    seen_urls.add(url)
                    print(f"Saved: {save_path}")
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")

                time.sleep(random.uniform(*DELAY_RANGE))

    print(f"Done: {downloaded} images downloaded to {style_dir}")

def run():
    for culture, queries in CULTURE_STYLES.items():
        style_dir = DATA_DIR / culture
        download_images(style_dir, queries, TARGET_PER_STYLE)

if __name__ == "__main__":
    run()
                