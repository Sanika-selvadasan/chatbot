import requests
from bs4 import BeautifulSoup
import hashlib

def scrape_website(url: str, max_paragraphs: int = 20) -> str:
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "lxml")
        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text(strip=True) for p in paragraphs[:max_paragraphs]])
        return content.strip()
    except Exception as e:
        return f"Error scraping site: {e}"

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
