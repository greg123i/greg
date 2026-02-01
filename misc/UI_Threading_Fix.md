# UI Threading & Sort Optimization

## Issue Resolved
The application was freezing (becoming unresponsive/ghosting) during the "Sort Data" operation. This was caused by running synchronous network requests (Ollama API) and file operations directly on the main UI thread.

## Solution Implemented

### 1. Threaded Execution
*   Moved the `classify_and_sort` operation to a background daemon thread (`threading.Thread`).
*   This ensures the main Pygame event loop continues to run (handling mouse, keyboard, and drawing) while sorting happens in the background.

### 2. Progress Feedback
*   Updated `lib/data_tools.py` to accept a `progress_callback`.
*   The worker thread reports progress (e.g., "Sorting file 5/100") back to the UI via this callback.
*   The UI updates the status bar in real-time.

### 3. Cancellation Support
*   Added a `threading.Event` (`sort_cancel_event`) to signal cancellation.
*   The sorting loop in `lib/data_tools.py` checks this event before processing each file.
*   The "Sort Data" button dynamically changes to "Stop Sorting" (Red) during operation. Clicking it triggers the cancellation event.

### 4. Robustness
*   Added a **5-second timeout** to Ollama API requests to prevent infinite hanging on network issues.
*   Implemented **Partial File Reading** (reading only first 2KB) to prevent memory spikes with large files.

## Usage for Developers

When adding new long-running tasks to the GUI:
1.  **Do not** call the function directly in the event handler.
2.  Define a worker function that updates state via `nonlocal` variables or shared state.
3.  Use `threading.Thread(target=worker, daemon=True).start()`.
4.  Pass a `threading.Event` if cancellation is required.

## Files Modified
*   `app/gui.py`: Added threading logic and UI state management.
*   `lib/data_tools.py`: Added callback/cancellation support and timeouts.
