import requests
from bs4 import BeautifulSoup
import hashlib

# def scrape_website(url: str, max_paragraphs: int = 20) -> str:
#     print(f'Inside scrape_website')
#     try:
#         response = requests.get(url, timeout=10)
#         soup = BeautifulSoup(response.text, "lxml")
#         paragraphs = soup.find_all("div")
#         content = "\n".join([p.get_text(strip=True) for p in paragraphs[:max_paragraphs]])
#         print(f'Scraped content\n{content}')
#         return content.strip()
#     except Exception as e:
#         return f"Error scraping site: {e}"

def scrape_website(url, max_paragraphs=15):
    try:
        response = requests.get(url, timeout=10)
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except Exception:
            soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        content = "\n".join([p.get_text(strip=True) for p in paragraphs[:max_paragraphs]])

        if not content.strip():
            content = soup.get_text()

        return content[:5000]
    except Exception as e:
        return f"Error scraping site: {e}"

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()