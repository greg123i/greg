import time
import urllib.request
import json
import statistics

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3" # Default model
TEST_PROMPTS = [
    "Hello",
    "Rate this text: " + "A" * 100,
    "Rate this text: " + "B" * 500, # Max length used in app
]

def check_model_availability(model_name):
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req) as r:
            tags = json.loads(r.read())
            models = [m['name'] for m in tags['models']]
            print(f"Available models: {models}")
            if model_name in models or f"{model_name}:latest" in models:
                return model_name if model_name in models else f"{model_name}:latest"
            # Fallback to first available if specific one missing
            if models:
                return models[0]
            return None
    except Exception as e:
        print(f"Error checking models: {e}")
        return None

def profile_request(model, prompt):
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    start = time.time()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps(data).encode('utf-8'))
        # Using a long timeout for profiling to catch the actual duration
        with urllib.request.urlopen(req, timeout=60) as r:
            _ = r.read()
        duration = time.time() - start
        return duration, None
    except Exception as e:
        duration = time.time() - start
        return duration, str(e)

def log(msg):
    print(msg, flush=True)
    with open("misc/profile_log.txt", "a") as f:
        f.write(msg + "\n")

def run_profile():
    with open("misc/profile_log.txt", "w") as f:
        f.write("Starting Profile\n")

    log("--- Ollama Timeout Analysis Profiler ---")
    
    # 1. Model Check
    # Force tinyllama for quick test if user wants, but we want to profile the REAL timeout cause which is likely llama3
    # So we keep llama3 as target if available
    target_model = "tinyllama"
    model = check_model_availability(target_model)
    
    if not model:
        log("No models found or Ollama not running.")
        return
    
    log(f"Target Model: {model}")
    log("-" * 30)

    # 2. Cold Start Test (First Request)
    log("Testing Cold Start / First Request...")
    duration, error = profile_request(model, "Warmup")
    if error:
        log(f"Cold Start Failed: {error} (Time: {duration:.2f}s)")
    else:
        log(f"Cold Start Time: {duration:.2f}s")
        if duration > 5.0:
            log("  -> WARNING: Cold start exceeds 5s default timeout!")

    # 3. Warm Performance Test
    log("\nTesting Warm Performance (5 requests)...")
    durations = []
    for i in range(5):
        d, err = profile_request(model, TEST_PROMPTS[2]) # Use 500 char prompt
        if err:
            log(f"  Req {i+1}: Failed ({err})")
        else:
            log(f"  Req {i+1}: {d:.2f}s")
            durations.append(d)
    
    if durations:
        avg = statistics.mean(durations)
        max_d = max(durations)
        log(f"\nWarm Avg: {avg:.2f}s")
        log(f"Warm Max: {max_d:.2f}s")
        
        if max_d > 5.0:
            log("  -> CRITICAL: Even warm requests can exceed 5s timeout.")
        elif avg > 2.0:
            log("  -> NOTE: Warm requests are slow, close to timeout margin.")
        else:
            log("  -> OK: Warm requests are well within 5s.")

if __name__ == "__main__":
    run_profile()
