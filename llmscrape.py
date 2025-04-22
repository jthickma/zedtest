import requests
from bs4 import BeautifulSoup
import pandas as pd
import google.generativeai as genai
import os
import time
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
START_URL = "https://example.com" # <--- CHANGE THIS TO THE TARGET URL
ALLOWED_DOMAIN = urlparse(START_URL).netloc # Stay within the same domain
MAX_PAGES_TO_CRAWL = 5  # Limit the number of pages to crawl
OUTPUT_FILE = "llm_analysis.txt"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REQUEST_DELAY = 1 # Seconds between requests to be polite

# --- Helper Functions ---

def fetch_page(url):
    """Fetches HTML content of a URL."""
    print(f"Fetching: {url}")
    try:
        # Add a user-agent to mimic a browser
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def parse_data(html, base_url):
    """Extracts text, links, image sources, and tables from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    data = {
        'text': "",
        'links': set(),
        'images': set(),
        'tables': []
    }

    # Extract text
    data['text'] = soup.get_text(separator=' ', strip=True)

    # Extract links
    for link in soup.find_all('a', href=True):
        href = link['href']
        full_url = urljoin(base_url, href)
        # Basic validation and ensure it's within the allowed domain
        parsed_link = urlparse(full_url)
        if parsed_link.scheme in ['http', 'https'] and parsed_link.netloc == ALLOWED_DOMAIN:
             data['links'].add(full_url)

    # Extract image sources
    for img in soup.find_all('img', src=True):
        src = img['src']
        full_img_url = urljoin(base_url, src)
        data['images'].add(full_img_url)

    # Extract tables using pandas
    try:
        tables_on_page = pd.read_html(html)
        for i, table_df in enumerate(tables_on_page):
             # Convert DataFrame to a simple string format (e.g., markdown)
             data['tables'].append(f"Table {i+1}:\n{table_df.to_markdown(index=False)}\n")
    except ValueError:
        # No tables found by pandas or parsing error
        pass
    except ImportError:
        print("Pandas dependency 'lxml' or 'html5lib' might be needed for table parsing.")
        pass


    return data

def call_llm(scraped_content):
    """Sends cleaned data to Google Generative AI and returns the interpretation."""
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found in environment variables."

    print("\nSending data to Generative AI...")
    genai.configure(api_key=GOOGLE_API_KEY)

    # Prepare the prompt - Summarize key content types
    # Be mindful of token limits. For very large sites, you might need to be selective.
    prompt = f"""Analyze the following data scraped from {START_URL} and its subpages. Provide a concise summary of the main topics, list key links/images, and summarize any tabular data found.

Extracted Text Summary (first 500 chars):
{scraped_content['all_text'][:500]}...

Key Links Found ({len(scraped_content['all_links'])} total):
{list(scraped_content['all_links'])[:10]} {'...' if len(scraped_content['all_links']) > 10 else ''}

Image Sources Found ({len(scraped_content['all_images'])} total):
{list(scraped_content['all_images'])[:10]} {'...' if len(scraped_content['all_images']) > 10 else ''}

Tables Found ({len(scraped_content['all_tables'])} total):
"""
    # Add table summaries
    for i, table_str in enumerate(scraped_content['all_tables']):
        if i >= 3: # Limit number of tables in prompt
             prompt += "... (more tables exist)\n"
             break
        prompt += f"\n--- Table {i+1} ---\n{table_str[:300]}...\n" # Limit table length in prompt


    try:
        model = genai.GenerativeModel('gemini-pro') # Or choose another appropriate model
        response = model.generate_content(prompt)
        print("LLM analysis received.")
        return response.text
    except Exception as e:
        print(f"Error calling Generative AI API: {e}")
        return f"Error during LLM processing: {e}"

def save_output(content, filename):
    """Saves content to a file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"LLM output saved to {filename}")
    except IOError as e:
        print(f"Error saving output to {filename}: {e}")

# --- Main Crawling Logic ---
def crawl():
    """Crawls the website starting from START_URL."""
    urls_to_visit = {START_URL}
    visited_urls = set()
    all_scraped_data = {
        'all_text': "",
        'all_links': set(),
        'all_images': set(),
        'all_tables': []
    }

    while urls_to_visit and len(visited_urls) < MAX_PAGES_TO_CRAWL:
        current_url = urls_to_visit.pop()

        if current_url in visited_urls:
            continue

        # Respect politeness delay
        time.sleep(REQUEST_DELAY)

        html_content = fetch_page(current_url)
        visited_urls.add(current_url)

        if html_content:
            page_data = parse_data(html_content, current_url)

            # Aggregate data (basic text concatenation, set union for links/images)
            # More sophisticated cleaning/aggregation might be needed
            all_scraped_data['all_text'] += page_data['text'] + "\n\n" # Add separators
            all_scraped_data['all_links'].update(page_data['links'])
            all_scraped_data['all_images'].update(page_data['images'])
            all_scraped_data['all_tables'].extend(page_data['tables'])

            # Add new, unvisited links to the queue
            new_links = page_data['links'] - visited_urls
            urls_to_visit.update(new_links)

    print(f"\nCrawling finished. Visited {len(visited_urls)} pages.")
    return all_scraped_data

# --- Execution ---
if __name__ == "__main__":
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please create a .env file with GOOGLE_API_KEY=YOUR_API_KEY_HERE")
    else:
        scraped_data = crawl()
        if scraped_data['all_text'] or scraped_data['all_links'] or scraped_data['all_images'] or scraped_data['all_tables']:
            llm_interpretation = call_llm(scraped_data)
            save_output(llm_interpretation, OUTPUT_FILE)
        else:
            print("No data was scraped successfully.")

