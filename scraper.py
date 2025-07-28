import requests
from bs4 import BeautifulSoup
import hashlib
import re

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

        # Try multiple approaches to get meaningful content
        content_parts = []
        
        # 1. Look for paragraphs
        paragraphs = soup.find_all('p')
        for p in paragraphs[:max_paragraphs]:
            text = p.get_text(strip=True)
            if len(text) > 20:  # Only include substantial paragraphs
                content_parts.append(text)
        
        # 2. Look for headings (h1, h2, h3)
        headings = soup.find_all(['h1', 'h2', 'h3'])
        for h in headings[:10]:
            text = h.get_text(strip=True)
            if len(text) > 5:
                content_parts.append(text)
        
        # 3. Look for divs with substantial text content
        divs = soup.find_all('div')
        for div in divs[:20]:
            text = div.get_text(strip=True)
            if len(text) > 50 and not re.match(r'^\s*$', text):
                # Filter out navigation, buttons, etc.
                if not any(skip in text.lower() for skip in ['cookie', 'privacy', 'terms', 'login', 'sign up', 'menu', 'navigation']):
                    content_parts.append(text)
        
        # 4. Look for spans with content
        spans = soup.find_all('span')
        for span in spans[:15]:
            text = span.get_text(strip=True)
            if len(text) > 20:
                content_parts.append(text)
        
        # Combine all content
        content = "\n".join(content_parts)
        
        # If still no content, try getting all text
        if not content.strip():
            content = soup.get_text()
            # Clean up the text
            content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with single space
            content = re.sub(r'\n\s*\n', '\n', content)  # Remove empty lines
        
        # Filter out very short or empty content
        if len(content.strip()) < 100:
            return f"Error: Insufficient content scraped from {url}. Content length: {len(content.strip())}"
        
        return content[:5000]
    except Exception as e:
        return f"Error scraping site: {e}"

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()