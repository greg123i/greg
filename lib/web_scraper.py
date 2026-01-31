import urllib.request
import urllib.robotparser
import urllib.parse
import time
import os
import random
import ssl
from html.parser import HTMLParser

# Ignore SSL certificate errors
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

class PageParser(HTMLParser):
    """
    HTML Parser to extract visible text AND links from a webpage.
    """
    def __init__(self, base_url):
        super().__init__()
        self.base_url = base_url
        self.extracted_text_list = []
        self.found_links = []
        self.is_inside_ignored_tag = False
        self.ignored_tags = ['script', 'style', 'nav', 'header', 'footer', 'noscript']

    def handle_starttag(self, tag, attrs):
        if tag in self.ignored_tags:
            self.is_inside_ignored_tag = True
            
        if tag == 'a':
            attrs_dict = dict(attrs)
            href = attrs_dict.get('href')
            if href:
                # Resolve relative URLs
                full_url = urllib.parse.urljoin(self.base_url, href)
                # Basic validation
                if full_url.startswith('http'):
                    self.found_links.append(full_url)

    def handle_endtag(self, tag):
        if tag in self.ignored_tags:
            self.is_inside_ignored_tag = False

    def handle_data(self, data):
        if not self.is_inside_ignored_tag:
            cleaned_data = data.strip()
            # Only keep substantial text chunks
            if cleaned_data and len(cleaned_data) > 20:
                self.extracted_text_list.append(cleaned_data)

    def get_text(self):
        return "\n".join(self.extracted_text_list)

class SearchParser(HTMLParser):
    """
    HTML Parser to extract search result links from DuckDuckGo HTML.
    """
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
            
        attributes_dictionary = dict(attrs)
        href_value = attributes_dictionary.get('href')
        
        if not href_value:
            return
            
        candidate_link = None
        
        # Handle DDG redirects (e.g., /l/?uddg=...)
        if '/l/?' in href_value:
            try:
                parsed_url = urllib.parse.urlparse(href_value)
                query_parameters = urllib.parse.parse_qs(parsed_url.query)
                if 'uddg' in query_parameters:
                    candidate_link = query_parameters['uddg'][0]
            except Exception:
                pass
        
        # Handle direct links - VERY PERMISSIVE
        elif href_value.startswith('http'):
            candidate_link = href_value
        
        if candidate_link:
            # Filter out obvious junk
            is_junk = False
            junk_domains = ['duckduckgo.com', 'google.com', 'bing.com']
            for domain in junk_domains:
                if domain in candidate_link:
                    is_junk = True
                    break
            
            if not is_junk:
                self.links.append(candidate_link)

class SafeScraper:
    """
    A respectful web scraper that honors robots.txt and uses rate limiting.
    """
    def __init__(self, output_directory):
        self.output_directory = output_directory
        self.robot_parser = urllib.robotparser.RobotFileParser()
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
    def can_fetch(self, url):
        """Check if robots.txt allows scraping this URL."""
        try:
            parsed_url = urllib.parse.urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            robots_txt_url = f"{base_url}/robots.txt"
            
            # Simple caching could go here, but for now just check
            self.robot_parser.set_url(robots_txt_url)
            self.robot_parser.read()
            return self.robot_parser.can_fetch(self.user_agent, url)
        except Exception as error:
            return True

    def scrape(self, url):
        """
        Scrapes text and links from a single URL if allowed.
        Returns: (extracted_text, found_links) or (None, [])
        """
        is_allowed = self.can_fetch(url)
        
        if not is_allowed:
            print(f"Skipping {url}: Disallowed by robots.txt")
            return None, []

        print(f"Scraping: {url}")
        try:
            request = urllib.request.Request(
                url, 
                headers={'User-Agent': self.user_agent}
            )
            
            with urllib.request.urlopen(request, timeout=15, context=ssl_context) as response:
                html_content = response.read().decode('utf-8', errors='ignore')
                
                parser = PageParser(url)
                parser.feed(html_content)
                extracted_text = parser.get_text()
                
                if len(extracted_text) < 100:
                    print(f"Skipping {url}: Not enough text found.")
                    return None, []

                # Save text
                safe_name_encoded = urllib.parse.quote(url, safe='')
                safe_name = safe_name_encoded[:50]
                filename = f"scraped_{safe_name}.txt"
                file_path = os.path.join(self.output_directory, filename)
                
                with open(file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(extracted_text)
                    
                print(f"Saved {len(extracted_text)} chars to {filename}")
                return extracted_text, parser.found_links
                
        except Exception as error:
            print(f"Error scraping {url}: {error}")
            return None, []

        # Rate limiting
        sleep_time = random.uniform(1, 3)
        time.sleep(sleep_time)

def search_web(query, num_results=20):
    """
    Searches DuckDuckGo (HTML version) for links.
    """
    print(f"Searching for: {query} (Target: {num_results} URLs)")
    collected_links = []
    seen_links = set()
    offset = 0
    
    while len(collected_links) < num_results:
        try:
            encoded_query = urllib.parse.quote(query)
            if offset == 0:
                url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            else:
                url = f"https://html.duckduckgo.com/html/?q={encoded_query}&s={offset}&dc={offset}"
            
            request = urllib.request.Request(
                url,
                headers={'User-Agent': "GregAIThingyScraper/1.0"}
            )
            
            with urllib.request.urlopen(request, timeout=15) as response:
                html_content = response.read().decode('utf-8', errors='ignore')
                
                parser = SearchParser()
                parser.feed(html_content)
                
                new_links_count = 0
                for link in parser.links:
                    if link not in seen_links:
                        collected_links.append(link)
                        seen_links.add(link)
                        new_links_count += 1
                        if len(collected_links) >= num_results:
                            break
                
                print(f"Page offset {offset}: Found {new_links_count} new links (Total: {len(collected_links)})")
                
                if new_links_count == 0:
                    print("No new links found on this page. Stopping search.")
                    break
                    
                offset += 30
                time.sleep(1)
                
        except Exception as error:
            print(f"Search failed at offset {offset}: {error}")
            break
            
    return collected_links[:num_results]

def get_fallback_urls():
    """Returns a list of fallback URLs to use if search fails."""
    return [
        "https://www.example.com",
        "https://www.gnu.org/home.en.html",
        "https://www.w3.org/",
        "https://httpbin.org/html",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.python.org/",
        "https://docs.python.org/3/tutorial/index.html",
        "https://scikit-learn.org/stable/",
        "https://pytorch.org/",
        "https://tensorflow.org/",
        "https://keras.io/",
        "https://openai.com/",
        "https://deepmind.google/",
        "https://www.ibm.com/topics/artificial-intelligence",
        "https://www.oracle.com/artificial-intelligence/what-is-ai/",
        "https://aws.amazon.com/machine-learning/",
        "https://azure.microsoft.com/en-us/overview/ai-platform/"
    ]
