import os
import math

def split_tinystories(input_path, output_dir, num_files=5000):
    """
    Splits the TinyStories dataset into `num_files` separate text files.
    """
    print(f"Reading {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by delimiter
    stories = content.split('<|endoftext|>')
    
    # Filter out empty strings and strip whitespace
    stories = [s.strip() for s in stories if s.strip()]
    
    total_stories = len(stories)
    print(f"Found {total_stories} stories.")
    
    if total_stories == 0:
        print("No stories found.")
        return

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}")
    else:
        print(f"Using directory {output_dir}")

    # Calculate stories per file
    # If we have fewer stories than requested files, limit num_files
    if total_stories < num_files:
        print(f"Warning: Only {total_stories} stories available. Creating {total_stories} files instead of {num_files}.")
        num_files = total_stories

    # Distribute stories evenly to get exactly num_files
    base_size = total_stories // num_files
    remainder = total_stories % num_files
    
    print(f"Splitting into {num_files} files (base size {base_size}, remainder distributed)...")

    current_idx = 0
    for i in range(num_files):
        # Determine size for this file
        # The first 'remainder' files get one extra story
        size = base_size + 1 if i < remainder else base_size
        
        if size == 0:
            break
            
        end_idx = current_idx + size
        chunk_stories = stories[current_idx:end_idx]
        current_idx = end_idx
        
        # Join back with delimiter (optional, but good for consistency if we want to read them back as stories)
        # Let's append <|endoftext|> to each story to maintain format.
        file_content = "\n<|endoftext|>\n".join(chunk_stories) + "\n<|endoftext|>\n"
        
        filename = f"part_{i:05d}.txt"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
            
        if (i + 1) % 500 == 0:
            print(f"Written {i + 1}/{num_files} files...")

    print("Done!")

if __name__ == "__main__":
    # Define paths
    INPUT_FILE = r"c:\Users\QWE\Desktop\greg\app\data\tinystories_valid.txt"
    OUTPUT_DIR = r"c:\Users\QWE\Desktop\greg\app\data\tinystories_split"
    
    split_tinystories(INPUT_FILE, OUTPUT_DIR, num_files=5000)
