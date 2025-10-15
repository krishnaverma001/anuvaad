# scripts/scrape_bbc_nepali.py

import requests
from bs4 import BeautifulSoup
import datetime
import os

def scrape_bbc_nepali():
    """
    Scrapes news articles from the BBC Nepali homepage and saves them to a file.
    """
    # The base URL for BBC Nepali news
    BASE_URL = "https://www.bbc.com"
    START_URL = f"{BASE_URL}/nepali"
    
    # Get the current date to create a unique filename
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_filename = f"bbc_nepali_articles_{current_date}.txt"
    
    # Ensure the output directory exists
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Starting scrape of {START_URL}")
    print(f"Saving data to: {output_path}")

    try:
        # 1. Fetch the main homepage
        main_page = requests.get(START_URL)
        main_page.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx) 
        
        main_soup = BeautifulSoup(main_page.content, "html.parser")
        
        # 2. Find all links that likely lead to articles
        # This is a bit of trial and error; we look for <a> tags with hrefs
        # that match the pattern of BBC articles.
        article_links = set() # Use a set to avoid duplicate links
        for a_tag in main_soup.find_all("a", href=True):
            href = a_tag['href']
            # We filter for links that look like internal news articles
            if href.startswith("/nepali/articles/"):
                full_url = f"{BASE_URL}{href}"
                article_links.add(full_url)

        print(f"Found {len(article_links)} unique article links.")

        # 3. Visit each article and extract its text
        all_article_text = []
        for i, link in enumerate(article_links):
            try:
                print(f"  Scraping ({i+1}/{len(article_links)}): {link}")
                article_page = requests.get(link)
                article_page.raise_for_status()
                
                article_soup = BeautifulSoup(article_page.content, "html.parser")
                
                # Find all paragraph tags (<p>) which usually contain the article text
                paragraphs = article_soup.find_all("p")
                
                article_text = "\n".join([p.get_text() for p in paragraphs])
                all_article_text.append(article_text)
                
            except requests.exceptions.RequestException as e:
                print(f"    Could not fetch article {link}: {e}")
            except Exception as e:
                print(f"    An error occurred while processing {link}: {e}")

        # 4. Save the collected text to a file
        with open(output_path, "w", encoding="utf-8") as f:
            # Separate articles with a clear delimiter
            f.write("\n\n--- NEW ARTICLE ---\n\n".join(all_article_text))
            
        print(f"\nScraping complete. All text saved to {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the main page {START_URL}: {e}")

if __name__ == "__main__":
    scrape_bbc_nepali()