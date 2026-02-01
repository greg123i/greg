import os
import time
import shutil
import concurrent.futures
import random
import string
import urllib.request
import json
import statistics

# Configuration
TEST_DIR = "temp_benchmark_data"
FILE_SIZES = {
    "small": (100, 1024),          # 1KB
    "medium": (10, 1024 * 1024),   # 1MB
    "large": (2, 10 * 1024 * 1024) # 10MB
}
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "tinyllama" # Use a small model for testing if available

def generate_random_text(size):
    """Generates random text of approx size bytes."""
    chars = string.ascii_letters + string.digits + " " * 10
    return ''.join(random.choices(chars, k=size))

def setup_data():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    
    files = []
    print(f"Generating test data in {TEST_DIR}...")
    for label, (count, size) in FILE_SIZES.items():
        for i in range(count):
            fname = os.path.join(TEST_DIR, f"{label}_{i}.txt")
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(generate_random_text(size))
            files.append(fname)
    return files

def benchmark_io_read_full(files):
    start = time.time()
    bytes_read = 0
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            bytes_read += len(content)
    duration = time.time() - start
    return duration, bytes_read

def benchmark_io_read_partial(files, chunk_size=1000):
    start = time.time()
    bytes_read = 0
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(chunk_size)
            bytes_read += len(content)
    duration = time.time() - start
    return duration, bytes_read

def mock_ollama_request(content):
    # Simulate network + inference latency (approx 50ms for tiny model or network overhead)
    time.sleep(0.05) 
    return "ok"

def check_ollama_alive():
    try:
        with urllib.request.urlopen("http://localhost:11434/") as response:
            return response.status == 200
    except:
        return False

def benchmark_ollama_serial(files, use_mock=False):
    start = time.time()
    processed = 0
    for fpath in files:
        # Simulate reading first
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(1000)
            
        if use_mock:
            mock_ollama_request(content)
        else:
            # Real request
            try:
                data = {"model": MODEL, "prompt": content[:100], "stream": False}
                req = urllib.request.Request(OLLAMA_URL, data=json.dumps(data).encode('utf-8'))
                with urllib.request.urlopen(req) as r:
                    pass
            except:
                pass # Ignore errors for benchmark
        processed += 1
    duration = time.time() - start
    return duration, processed

def benchmark_ollama_threaded(files, max_workers=4, use_mock=False):
    start = time.time()
    processed = 0
    
    def task(fpath):
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(1000)
        
        if use_mock:
            mock_ollama_request(content)
        else:
            try:
                data = {"model": MODEL, "prompt": content[:100], "stream": False}
                req = urllib.request.Request(OLLAMA_URL, data=json.dumps(data).encode('utf-8'))
                with urllib.request.urlopen(req) as r:
                    pass
            except:
                pass
        return 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(task, files))
        processed = sum(results)
        
    duration = time.time() - start
    return duration, processed

def run_benchmarks():
    files = setup_data()
    total_files = len(files)
    print(f"Created {total_files} files.")
    
    print("\n--- I/O Benchmarks ---")
    t_full, b_full = benchmark_io_read_full(files)
    print(f"Full Read: {t_full:.4f}s ({b_full/1024/1024:.2f} MB)")
    
    t_part, b_part = benchmark_io_read_partial(files)
    print(f"Partial Read (1KB): {t_part:.4f}s ({b_part/1024/1024:.2f} MB)")
    
    improvement = (t_full - t_part) / t_full * 100
    print(f"I/O Improvement: {improvement:.1f}%")

    print("\n--- API/Inference Benchmarks ---")
    ollama_live = check_ollama_alive()
    print(f"Ollama Live: {ollama_live}")
    use_mock = not ollama_live
    
    if not ollama_live:
        print("Using Mock Inference (50ms latency)...")
    
    print("Running Serial Processing...")
    t_ser, _ = benchmark_ollama_serial(files, use_mock=use_mock)
    print(f"Serial Time: {t_ser:.4f}s (Avg: {t_ser/total_files*1000:.1f}ms/file)")
    
    print("Running Threaded Processing (4 workers)...")
    t_par, _ = benchmark_ollama_threaded(files, max_workers=4, use_mock=use_mock)
    print(f"Threaded Time: {t_par:.4f}s (Avg: {t_par/total_files*1000:.1f}ms/file)")
    
    speedup = t_ser / t_par
    print(f"Concurrency Speedup: {speedup:.2f}x")
    
    # Cleanup
    shutil.rmtree(TEST_DIR)

if __name__ == "__main__":
    run_benchmarks()
