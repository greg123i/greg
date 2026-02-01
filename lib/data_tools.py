import urllib.request
import urllib.error
import json
import os
import random
import subprocess
import time
import re

# Common Windows installation path for Ollama
OLLAMA_EXE = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")

from lib.web_scraper import SafeScraper, search_web

def crawl_and_scrape(data_dir, target_count=10000, seeds=None):
    """
    Crawls and scrapes text data from the web using search seeds and SafeScraper.
    """
    os.makedirs(data_dir, exist_ok=True)
    scraper = SafeScraper(data_dir)
    
    if seeds is None:
        # Varied topics for initial search
        seeds = [
            "Introduction to Philosophy", "Latest Technology News", "History of the World",
            "Learn Python Programming", "Creative Writing Tips", "Understanding Quantum Physics",
            "Best Cooking Recipes", "Travel Guides 2024", "Classic Literature summaries",
            "DIY Home Improvement"
        ]
    
    visited = set()
    to_visit = []
    
    # 1. Initialize queue with search results
    print("Initializing crawl with search seeds...")
    for query in seeds:
        print(f"Searching for: {query}")
        links = search_web(query, num_results=5) # Get top 5 for each topic
        for link in links:
            if link not in visited:
                to_visit.append(link)
                
    # Shuffle to mix topics
    random.shuffle(to_visit)
    
    count = len([f for f in os.listdir(data_dir) if f.endswith(".txt")])
    print(f"Starting crawl. Already have {count} files. Target: {target_count}. Queue size: {len(to_visit)}")
    
    while count < target_count and to_visit:
        url = to_visit.pop(0)
        if url in visited: continue
        visited.add(url)
        
        # Scrape
        text, new_links = scraper.scrape(url)
        
        if text:
            count += 1
            if count % 10 == 0:
                print(f"Scraped {count}/{target_count} files...")
                
        # Add new links to queue (DFS/BFS hybrid - append to end)
        # To maintain variety, maybe add only a subset or shuffle?
        # Let's just add them.
        if len(to_visit) < 500: # Limit queue size to prevent memory explosion
            for link in new_links:
                if link not in visited and link not in to_visit:
                    to_visit.append(link)
                    
        # Random shuffle occasionally to prevent getting stuck in one domain
        if count % 5 == 0:
            random.shuffle(to_visit)
            
    print(f"Crawl finished. Total files: {count}")
    return count >= target_count

def download_tinystories_subset(output_path, max_size_mb=5):
    """
    Downloads a small subset of TinyStories validation set.
    """
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
    
    print(f"Downloading subset from {url}...")
    
    try:
        # Stream download and stop after max_size_mb
        with urllib.request.urlopen(url) as response:
            with open(output_path, 'wb') as f:
                downloaded = 0
                block_size = 1024 * 64
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    if downloaded > max_size_mb * 1024 * 1024:
                        break
        
        print(f"Downloaded {downloaded / (1024*1024):.2f} MB to {output_path}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def download_tinystories_train(output_path, max_size_mb=500):
    """
    Downloads a larger subset of TinyStories TRAINING set.
    """
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    
    print(f"Downloading training data from {url}...")
    print(f"Target size: {max_size_mb} MB")
    
    try:
        # Stream download and stop after max_size_mb
        with urllib.request.urlopen(url) as response:
            with open(output_path, 'wb') as f:
                downloaded = 0
                block_size = 1024 * 64
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    
                    # Print progress every 10MB
                    if downloaded % (10 * 1024 * 1024) < block_size:
                         print(f"Downloaded {downloaded / (1024*1024):.1f} MB...", end='\r')

                    if downloaded > max_size_mb * 1024 * 1024:
                        break
        
        print(f"\nDownloaded {downloaded / (1024*1024):.2f} MB to {output_path}")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

def generate_ollama_data(output_path, model="tinyllama", count=10):
    """
    Generates diverse data using local Ollama instance.
    """
    print(f"Generating {count} items using Ollama ({model})...")
    
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    prompts = [
        "Write a short story about a robot learning to love.",
        "Explain quantum entanglement like I'm 5.",
        "Write a python function to calculate fibonacci numbers.",
        "What is the capital of France? Explain its history briefly.",
        "Write a haiku about coding.",
        "Describe a futuristic city.",
        "Why is the sky blue?",
        "Write a dialogue between a wizard and a programmer.",
        "List 5 healthy snacks.",
        "Explain the concept of recursion."
    ]
    
    generated_count = 0
    
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            for i in range(count):
                # Pick a prompt (cycle through or random)
                prompt = prompts[i % len(prompts)]
                
                data = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
                
                req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
                
                try:
                    with urllib.request.urlopen(req) as response:
                        result = json.loads(response.read().decode('utf-8'))
                        text = result.get('response', '')
                        
                        # Clean up text
                        text = text.replace('<|endoftext|>', '') # Ensure we don't save this
                        text = text.replace('\n', ' ').strip()
                        if text:
                            f.write(f"Q: {prompt} A: {text}\n\n")
                            generated_count += 1
                            print(f"Generated {generated_count}/{count}")
                        
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        print(f"Model '{model}' not found. Attempting to pull...")
                        if os.path.exists(OLLAMA_EXE):
                            try:
                                print(f"Pulling {model}...")
                                subprocess.run([OLLAMA_EXE, "pull", model], creationflags=subprocess.CREATE_NO_WINDOW, check=True)
                                print("Pull complete. Retrying generation...")
                                
                                # Retry the request
                                with urllib.request.urlopen(req) as response:
                                    result = json.loads(response.read().decode('utf-8'))
                                    text = result.get('response', '')
                                    text = text.replace('<|endoftext|>', '')
                                    text = text.replace('\n', ' ').strip()
                                    if text:
                                        f.write(f"Q: {prompt} A: {text}\n\n")
                                        generated_count += 1
                                        print(f"Generated {generated_count}/{count}")
                            except Exception as pull_err:
                                print(f"Failed to pull model: {pull_err}")
                        else:
                            print(f"Ollama executable not found at {OLLAMA_EXE}")
                    else:
                        print(f"HTTP Error: {e.code}")

                except urllib.error.URLError:
                    print(f"Ollama connection failed. Checking if we can auto-start...")
                    
                    if os.path.exists(OLLAMA_EXE):
                        print(f"Found Ollama at {OLLAMA_EXE}. Starting server...")
                        try:
                            # Start Ollama in background
                            subprocess.Popen([OLLAMA_EXE, "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
                            print("Waiting 5 seconds for Ollama to initialize...")
                            time.sleep(5)
                            
                            # Retry the request once
                            try:
                                with urllib.request.urlopen(req) as response:
                                    result = json.loads(response.read().decode('utf-8'))
                                    text = result.get('response', '')
                                    
                                    # Clean up text
                                    text = text.replace('<|endoftext|>', '')
                                    text = text.replace('\n', ' ').strip()
                                    if text:
                                        f.write(f"Q: {prompt} A: {text}\n\n")
                                        generated_count += 1
                                        print(f"Generated {generated_count}/{count}")
                            except urllib.error.HTTPError as http_err:
                                if http_err.code == 404:
                                    print(f"Model '{model}' not found after start. Pulling...")
                                    subprocess.run([OLLAMA_EXE, "pull", model], creationflags=subprocess.CREATE_NO_WINDOW, check=True)
                                    # Retry generation again
                                    with urllib.request.urlopen(req) as response:
                                        result = json.loads(response.read().decode('utf-8'))
                                        text = result.get('response', '')
                                        text = text.replace('<|endoftext|>', '')
                                        text = text.replace('\n', ' ').strip()
                                        if text:
                                            f.write(f"Q: {prompt} A: {text}\n\n")
                                            generated_count += 1
                                            print(f"Generated {generated_count}/{count}")
                                else:
                                    raise http_err
                            except Exception as retry_err:
                                print(f"Retry failed: {retry_err}")
                                    
                        except Exception as e:
                            print(f"Auto-start/Auto-pull failed: {e}")
                            return False
                    else:
                        print(f"Ollama not found at {OLLAMA_EXE}. Please install it or run 'ollama serve' manually.")
                        return False
                    
        return True
    except Exception as e:
        print(f"Generation failed: {e}")
        return False


