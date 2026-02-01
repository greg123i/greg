import urllib.request
import json
import time

url = "http://localhost:11434/api/generate"
data = {"model": "tinyllama", "prompt": "hi", "stream": False}

print("Sending request...", flush=True)
try:
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'))
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f"Response: {r.status}", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
print("Done.", flush=True)
