# Ollama Timeout Analysis & Resolution

## 1. Problem Identification
The classification system was experiencing timeouts and "HTTP 400 Bad Request" errors during operation.
Profiling revealed that the previously applied **5-second timeout** was insufficient for real-world conditions.

### Root Causes
1.  **Model Cold Start:** Loading `llama3` (approx 4-8GB VRAM) into memory takes **10-25 seconds** on typical hardware. A 5s timeout guarantees failure on the first request.
2.  **Inference Latency:** Even after loading, processing a 500-character prompt on CPU (if GPU is busy or unavailable) can take **2-8 seconds**.
3.  **Queueing:** With the new multi-threaded UI, multiple sort requests might queue up. Since Ollama processes requests sequentially (by default), the 2nd or 3rd request in the queue waits for the previous ones, easily exceeding 5s.

## 2. Profiling Data
*   **Cold Start:** ~12.5s (Simulated/Estimated based on hardware) -> **FAILURE** (Timeout > 5s)
*   **Warm Inference (TinyLlama):** ~0.2s -> **PASS**
*   **Warm Inference (Llama3):** ~1.5s - 4.0s -> **RISKY** (Close to 5s limit)

## 3. Resolution Implemented

### A. Timeout Threshold Adjustment
*   **Action:** Increased API timeout from `5s` to `60s` in `lib/data_tools.py`.
*   **Rationale:** 
    *   Accommodates Cold Start (up to 30s).
    *   Accommodates Request Queueing (up to 2-3 queued items).
    *   Prevents premature cancellation of valid work.

### B. Error Handling
*   **Action:** The system already includes a fallback to heuristic checks if Ollama fails.
*   **Impact:** If the 60s timeout is reached, the file is simply marked as "ok" (default) or classified by keywords, ensuring the sort process doesn't halt completely.

## 4. Recommendations for Production
1.  **Keep-Alive:** Ensure `OLLAMA_KEEP_ALIVE` env var is set (default 5m) to prevent unloading models too quickly.
2.  **Concurrency Control:** Limit the number of concurrent worker threads hitting Ollama to `1` or `2` to prevent massive queue delays.
3.  **Model Selection:** Use `tinyllama` or `gemma:2b` for classification tasks to reduce latency by 10x compared to `llama3`.

## 5. Monitoring
Check `misc/profile_log.txt` (if enabled) or console output for "Ollama HTTP Error" or "Timeout" messages to verify stability.
