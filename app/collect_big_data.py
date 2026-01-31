import os
import sys

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lib.data_tools import download_tinystories_subset, download_tinystories_train

def collect_and_split_data(target_count=5000, target_length=5000):
    """
    Downloads a large corpus and splits it into specific file counts and lengths.
    """
    data_dir = os.path.join(PROJECT_ROOT, "app", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Calculate required size in MB
    total_bytes = target_count * target_length
    required_mb = (total_bytes / (1024 * 1024)) + 5 # 5MB buffer
    
    print(f"Goal: {target_count} files of {target_length} characters each.")
    print(f"Total data needed: {required_mb:.2f} MB")
    
    temp_file = os.path.join(data_dir, "temp_corpus.txt")
    
    # 1. Download base data
    if not os.path.exists(temp_file) or os.path.getsize(temp_file) < total_bytes:
        # Use training set if we need more than 30MB, otherwise val set is fine
        if required_mb > 30:
            print("Downloading from Training set (larger)...")
            success = download_tinystories_train(temp_file, max_size_mb=int(required_mb))
        else:
            print("Downloading from Validation set (smaller)...")
            success = download_tinystories_subset(temp_file, max_size_mb=int(required_mb))
            
        if not success:
            print("Failed to download corpus.")
            return

    # 2. Split into files
    print("Splitting corpus into files...")
    with open(temp_file, 'rb') as f:
        corpus = f.read()
    
    # Clean up corpus a bit (ensure it's valid ASCII/UTF-8 for the brain)
    # We keep newlines as requested
    
    current_pos = 0
    total_len = len(corpus)
    files_created = 0
    
    for i in range(target_count):
        if current_pos + target_length > total_len:
            print(f"Corpus too small. Only created {files_created} files.")
            break
            
        chunk = corpus[current_pos : current_pos + target_length]
        
        # Save as individual file
        file_name = f"greg_data_{i:04d}.txt"
        file_path = os.path.join(data_dir, file_name)
        
        with open(file_path, 'wb') as f_out:
            f_out.write(chunk)
            
        current_pos += target_length
        files_created += 1
        
        if files_created % 500 == 0:
            print(f"Created {files_created}/{target_count} files...")

    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    print(f"Successfully created {files_created} training files in {data_dir}")

if __name__ == "__main__":
    # Run with defaults or args
    count = 5000
    length = 5000
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    if len(sys.argv) > 2:
        length = int(sys.argv[2])
        
    collect_and_split_data(count, length)
