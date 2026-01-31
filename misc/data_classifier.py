import os
import shutil
import subprocess
import time

# Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'data'))
CLASSES = ["really_bad", "bad", "ok", "good", "really_good"]
OLLAMA_MODEL = "llama3" # Or whatever model is available, user said "use ollama"

def ensure_folders():
    for cls in CLASSES:
        path = os.path.join(DATA_DIR, cls)
        os.makedirs(path, exist_ok=True)

def get_ollama_rating(text_sample):
    """
    Asks Ollama to rate the text.
    """
    prompt = f"""
    Analyze the quality of the following text for training an AI model.
    Rate it as one of the following categories exactly: really bad, bad, ok, good, really good.
    
    Criteria:
    - really bad: Gibberish, random characters, mostly empty, or decoding errors.
    - bad: Poor grammar, incoherent, spam, or very repetitive.
    - ok: Readable but generic, potential minor errors, or average quality.
    - good: Clear, coherent, informative, well-structured.
    - really good: Exceptional quality, educational, high-value content (e.g. textbooks, high-quality code, classic literature).

    Output ONLY the category name in lowercase.

    Text Sample (first 500 chars):
    {text_sample[:500]}
    """
    
    try:
        # Run Ollama
        # Note: 'ollama run llama3 "prompt"' might be slow if it reloads model every time.
        # Ideally use API, but subprocess is requested/implied context.
        # Assuming 'ollama' is in PATH.
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        output = result.stdout.strip().lower()
        
        # Clean up output to find the category
        for cls in CLASSES:
            if cls.replace('_', ' ') in output:
                return cls
        
        # Fallback if model chats
        return "ok" 
    except Exception as e:
        print(f"Ollama Error: {e}")
        return None

def process_files():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"Found {len(files)} files to classify in {DATA_DIR}")
    
    for i, fname in enumerate(files):
        fpath = os.path.join(DATA_DIR, fname)
        
        # Read sample
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
            
        if len(content.strip()) < 10:
            rating = "really_bad"
        else:
            rating = get_ollama_rating(content)
            
        if rating:
            target_dir = os.path.join(DATA_DIR, rating)
            target_path = os.path.join(target_dir, fname)
            
            print(f"[{i+1}/{len(files)}] {fname} -> {rating}")
            try:
                shutil.move(fpath, target_path)
            except Exception as e:
                print(f"Error moving file: {e}")
        else:
            print(f"[{i+1}/{len(files)}] {fname} -> Failed to rate")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
    else:
        ensure_folders()
        process_files()
