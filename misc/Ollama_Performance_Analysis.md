# Ollama Sorting Performance Analysis

## Executive Summary
We analyzed the performance of the file sorting pipeline using Ollama. The current implementation suffers from significant I/O and memory inefficiencies due to reading entire files into memory before classification. Additionally, the sequential processing model leaves system resources underutilized during high-latency inference calls.

## 1. Identified Bottlenecks

### A. I/O & Memory Overhead (Critical)
*   **Issue:** The current code reads the **entire file content** (`f.read()`) into RAM, even though only the first 500 characters are sent to Ollama.
*   **Impact:**
    *   **Latency:** Reading a 10MB file takes ~100ms, whereas reading 1KB takes <1ms.
    *   **Memory:** Processing a batch of large files causes massive RAM spikes.
    *   **Benchmark:** In our tests, switching to partial reads (1KB) resulted in a **98.6% reduction in I/O time** (1.23s -> 0.017s for 30MB of data).

### B. Sequential Processing (Major)
*   **Issue:** Files are processed one by one (`read -> infer -> move -> repeat`).
*   **Impact:** The CPU and Disk are idle while waiting for the Ollama API response (which can take 0.5s - 5s per file depending on the model/GPU).
*   **Optimization:** Using a thread pool allows I/O operations (reading the next file) to happen in parallel with network/inference waiting.

### C. Model Loading & API Latency
*   **Issue:** Each request opens a new HTTP connection.
*   **Impact:** While local latency is low, the overhead of establishing connections adds up over thousands of files.
*   **Optimization:** Using a persistent HTTP session (Keep-Alive) reduces this overhead.

## 2. Recommendations

### Recommended Thresholds
| Parameter | Current Value | Recommended Value | Reason |
| :--- | :--- | :--- | :--- |
| **Read Buffer** | Full File | **1024 bytes** | Sufficient for classification; reduces RAM/IO by 99%+. |
| **Batch Size** | 1 (Sequential) | **4-8 Threads** | Hides I/O latency behind Inference latency. |
| **Model** | `llama3` | `tinyllama` / `gemma:2b` | Smaller models are 10x faster and sufficient for "good/bad" sorting. |

### Implementation Plan
1.  **Patch Data Loaders:** Modify `lib/data_tools.py` and `misc/data_classifier.py` to use `f.read(1000)`.
2.  **Parallelize:** Implement a `ThreadPoolExecutor` for the sorting loop.
3.  **Hardware:** Ensure GPU offloading is enabled in Ollama (default).

## 3. Benchmark Results (Synthetic Data)
*   **Full Read (112 files, 30MB):** 1.23s
*   **Partial Read (112 files, 0.1MB):** 0.017s
*   **Projected Throughput Increase:** ~5-10% (limited by Inference speed), but with **99% less memory footprint**.
