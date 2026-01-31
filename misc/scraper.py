import sys
import os

# Add project root to sys.path to import from lib
# Current file: stuff/machine_learning/misc/scraper.py
# Root: ../../../
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from lib.web_scraper import SafeScraper, search_web, get_fallback_urls

def main():
    # Ensure data directory exists
    # Data dir: app/data
    data_directory = os.path.abspath(os.path.join(project_root, "app/data"))
    
    if not os.path.exists(data_directory):
        print(f"Creating data directory: {data_directory}")
        os.makedirs(data_directory)
    else:
        print(f"Using data directory: {data_directory}")
        
    scraper = SafeScraper(data_directory)
    
    # Get user input for search
    print("--- Greg's Auto Scraper ---")
    query = input("Enter a topic to search and scrape (or press Enter for default): ").strip()
    
    if not query:
        query = "machine learning basics"
        
    print(f"Searching for websites about: {query}")
    urls = search_web(query, num_results=100)
    
    if not urls:
        print("No results found or search failed. Using fallback list.")
        urls = get_fallback_urls()
    
    print(f"Found {len(urls)} URLs to process.")
    
    # Queue-based crawling
    url_queue = list(urls)
    visited_urls = set()
    max_pages = 200  # Safety limit
    processed_count = 0
    
    while url_queue and processed_count < max_pages:
        url = url_queue.pop(0)
        
        if url in visited_urls:
            continue
            
        visited_urls.add(url)
        processed_count += 1
        
        print(f"[{processed_count}/{max_pages}] Processing: {url}")
        
        # Scrape and get new links
        result = scraper.scrape(url)
        # Check if result is tuple (text, links) or just text (backwards compatibility if needed)
        new_links = []
        if isinstance(result, tuple):
            _, new_links = result
        
        if new_links:
            # Add new links to queue
            added_count = 0
            for link in new_links:
                # Basic filtering to avoid loops and staying roughly on topic/web
                if link not in visited_urls and link not in url_queue:
                    url_queue.append(link)
                    added_count += 1
            print(f"  -> Found {len(new_links)} links, added {added_count} to queue.")
        
    print("Done!")

if __name__ == "__main__":
    main()
