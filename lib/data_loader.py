import os
import random
import numpy as np

class FileAnalyzer:
    """
    Handles file vectorization and clustering for diverse training.
    """
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.file_vectors = {} # file_path -> 8D vector
        self.clusters = {}     # cluster_id -> list of file_paths
        
        # Fixed 8 characters for vectorization (common English chars + space)
        self.target_chars = [32, 101, 116, 97, 111, 105, 110, 115] # ' ', 'e', 't', 'a', 'o', 'i', 'n', 's'

    def vectorize_files(self):
        """
        Generates 8D vectors for all .txt files based on char frequency.
        """
        print("Vectorizing files...")
        files = []
        for root, dirs, f_names in os.walk(self.data_directory):
            if 'really_bad' in dirs:
                dirs.remove('really_bad')
            for f in f_names:
                if f.endswith(".txt"):
                    files.append(os.path.join(root, f))
        
        for fpath in files:
            try:
                with open(fpath, 'rb') as f:
                    content = f.read()
                    if not content: continue
                    
                    vector = []
                    total_len = len(content)
                    for char_code in self.target_chars:
                        count = content.count(char_code)
                        vector.append(count / total_len)
                    
                    self.file_vectors[fpath] = np.array(vector)
            except Exception as e:
                print(f"Error vectorizing {fpath}: {e}")
        
        print(f"Vectorized {len(self.file_vectors)} files.")

    def cluster_files(self, k=10):
        """
        Groups files into k clusters using a simple K-Means implementation.
        """
        if not self.file_vectors:
            return
            
        print(f"Clustering files into {k} groups...")
        paths = list(self.file_vectors.keys())
        vectors = np.array([self.file_vectors[p] for p in paths])
        
        # Initialize centroids randomly from existing vectors
        centroids = vectors[np.random.choice(len(vectors), k, replace=False)]
        
        for _ in range(10): # 10 iterations is enough for this
            # Assign clusters
            new_clusters = {i: [] for i in range(k)}
            for i, v in enumerate(vectors):
                distances = np.linalg.norm(centroids - v, axis=1)
                cluster_id = np.argmin(distances)
                new_clusters[cluster_id].append(paths[i])
            
            # Update centroids
            for i in range(k):
                if new_clusters[i]:
                    cluster_vectors = np.array([self.file_vectors[p] for p in new_clusters[i]])
                    centroids[i] = cluster_vectors.mean(axis=0)
            
            self.clusters = new_clusters
        
        print("Clustering complete.")
        for cid, files in self.clusters.items():
            print(f"  Cluster {cid}: {len(files)} files")

    def get_diverse_sample(self, n=10):
        """
        Samples files from as many different clusters as possible.
        """
        if not self.clusters:
            return list(self.file_vectors.keys())[:n]
            
        sample = []
        cluster_ids = list(self.clusters.keys())
        random.shuffle(cluster_ids)
        
        while len(sample) < n:
            for cid in cluster_ids:
                if self.clusters[cid]:
                    f = random.choice(self.clusters[cid])
                    if f not in sample:
                        sample.append(f)
                if len(sample) >= n: break
            if all(not self.clusters[cid] for cid in cluster_ids): break
            
        return sample

def load_text_data(samples=1000, sequence_length=10, target_length=1, data_directory=None):
    """
    Loads text data from the 'data' directory and prepares it for training.
    
    Args:
        samples (int): Number of sequences to generate.
        sequence_length (int): Length of each sequence.
        target_length (int): Length of target sequence.
        data_directory (str): Path to data directory. If None, tries to find it.
        
    Returns:
        tuple: (X, y) where X is input sequences and y is target values.
    """
    input_sequences = []
    target_values = []
    
    if data_directory is None:
        # data directory is at project root/data
        # c:\Users\QWE\Desktop\greg\lib\..\data
        base_dir = os.path.dirname(__file__)
        data_directory = os.path.abspath(os.path.join(base_dir, '..', 'data'))
    
    all_byte_content = bytearray()
    
    # Read all text files recursively
    if os.path.exists(data_directory):
        for root, dirs, files in os.walk(data_directory):
            # Skip 'really_bad' folder
            if 'really_bad' in dirs:
                dirs.remove('really_bad')
                
            for filename in files:
                if filename.endswith(".txt"):
                    try:
                        file_path = os.path.join(root, filename)
                        # Read as binary to get raw bytes (0-255)
                        with open(file_path, 'rb') as file_handle:
                            content_bytes = file_handle.read()
                            
                            # Filter out unwanted tokens (in bytes)
                            # <|endoftext|> is b'<|endoftext|>'
                            content_bytes = content_bytes.replace(b"<|endoftext|>", b"")
                            content_bytes = content_bytes.replace(b"<|endoftext", b"")
                            
                            # Add to collection (space separator is byte 32)
                            all_byte_content.extend(content_bytes)
                            all_byte_content.append(32) 
                            
                    except Exception as error:
                        print(f"Error reading {filename}: {error}")
    
    if len(all_byte_content) < sequence_length + target_length:
        print("Not enough scraped data. Using fallback patterns.")
        return generate_text_data_fallback(samples, sequence_length, target_length)
        
    print(f"Loaded {len(all_byte_content)} bytes of data.")
    
    # Create sequences
    # Randomly sample sequences
    all_bytes_arr = np.array(all_byte_content, dtype=np.uint8)
    
    for _ in range(samples):
        # Calculate max starting index
        total_length = sequence_length + target_length
        max_start_index = len(all_bytes_arr) - total_length
        start_index = random.randint(0, max_start_index)
        
        sequence = all_bytes_arr[start_index : start_index + total_length]
        
        # Already 0-255 uint8
        input_part = sequence[:sequence_length]
        target_part = sequence[sequence_length:]
        
        input_sequences.append(input_part)
        target_values.append(target_part)
        
    return np.array(input_sequences), np.array(target_values)

def load_raw_byte_data(data_directory=None):
    """
    Loads the full raw content as bytes from the data directory.
    Returns a list of integers (0-255).
    """
    if data_directory is None:
        base_dir = os.path.dirname(__file__)
        data_directory = os.path.abspath(os.path.join(base_dir, '..', 'data'))
    
    all_byte_content = bytearray()
    
    # Read all text files recursively
    if os.path.exists(data_directory):
        for root, dirs, files in os.walk(data_directory):
            if 'really_bad' in dirs:
                dirs.remove('really_bad')
                
            for filename in files:
                if filename.endswith(".txt"):
                    try:
                        file_path = os.path.join(root, filename)
                        with open(file_path, 'rb') as file_handle:
                            content_bytes = file_handle.read()
                            
                            # Filter out unwanted tokens (in bytes)
                            content_bytes = content_bytes.replace(b"<|endoftext|>", b"")
                            content_bytes = content_bytes.replace(b"<|endoftext", b"")
                            
                            all_byte_content.extend(content_bytes)
                            all_byte_content.append(32) # Space separator
                            
                    except Exception as error:
                        print(f"Error reading {filename}: {error}")
    
    return list(all_byte_content)

def load_raw_text_data(data_directory=None):
    """
    Loads the full raw text content from the data directory.
    Returns a single numpy array of ASCII values.
    """
    if data_directory is None:
        base_dir = os.path.dirname(__file__)
        data_directory = os.path.abspath(os.path.join(base_dir, '..', 'data'))
    
    all_text_content = ""
    
    # Read all text files recursively
    if os.path.exists(data_directory):
        for root, dirs, files in os.walk(data_directory):
            if 'really_bad' in dirs:
                dirs.remove('really_bad')

            for filename in files:
                if filename.endswith(".txt"):
                    try:
                        file_path = os.path.join(root, filename)
                        with open(file_path, 'r', encoding='utf-8') as file_handle:
                            content = file_handle.read()
                            
                            # Filter out unwanted tokens
                            content = content.replace("<|endoftext|>", "")
                            content = content.replace("<|endoftext", "")
                            
                            content_parts = content.split()
                            cleaned_content = " ".join(content_parts)
                            all_text_content += cleaned_content + " "
                    except Exception as error:
                        print(f"Error reading {filename}: {error}")
    
    if not all_text_content:
        # Fallback
        all_text_content = "abcdefghijklmnopqrstuvwxyz " * 1000
        
    # Convert to ASCII array
    ascii_vals = [ord(c) for c in all_text_content]
    return np.array(ascii_vals, dtype=np.float32)

def generate_text_data_fallback(samples=1000, sequence_length=10, target_length=1):
    """
    Generates text sequence data (Fallback if no files).
    Patterns: Linear (abc...), Repeat (abab...), Reverse (zyx...)
    """
    input_sequences = []
    target_values = []
    
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    
    for _ in range(samples):
        pattern_type = random.choice(['linear', 'repeat', 'reverse'])
        
        # Ensure we have enough room for the sequence
        total_length = sequence_length + target_length
        # Ensure start index is valid
        max_start_index = max(0, len(alphabet) - total_length - 1)
        start_index = random.randint(0, max_start_index)
        
        sequence_characters = ""
        
        if pattern_type == 'linear':
            # abcde...
            # Ensure we don't go out of bounds if alphabet is too short for requested length
            full_slice = alphabet[start_index:]
            while len(full_slice) < total_length:
                full_slice += alphabet # Wrap around if needed
            sequence_characters = full_slice[:total_length]
            
        elif pattern_type == 'repeat':
            # ababab...
            # Pick 2 or 3 chars to repeat
            step = random.randint(2, 3)
            substring = alphabet[start_index : start_index + step]
            if len(substring) == 0: substring = "ab"
            # Repeat enough times
            repeated_string = substring * (total_length)
            sequence_characters = repeated_string[:total_length]
            
        elif pattern_type == 'reverse':
             # edcba...
            substring = alphabet[start_index : start_index + total_length]
            # Wrap around handling
            while len(substring) < total_length:
                substring += alphabet
            substring = substring[:total_length]
            sequence_characters = substring[::-1]
        else:
            # Fallback
            sequence_characters = "a" * total_length
            
        # Convert to numbers (ASCII)
        sequence_numbers = []
        for char in sequence_characters:
            sequence_numbers.append(ord(char))
        
        input_part = sequence_numbers[:sequence_length]
        target_part = sequence_numbers[sequence_length:]
        
        input_sequences.append(input_part)
        target_values.append(target_part)
        
    return np.array(input_sequences), np.array(target_values)
