# Greg (The Organism-like AI)

Greg is a custom, modular Artificial Intelligence designed with an "organism-like" architecture rather than a standard monolithic Transformer. It separates perception (Reading), cognition (Thinking), and action (Writing) into distinct, interacting sub-brains.

## Architecture

Unlike standard LLMs that process tokens in a fixed sequence, Greg uses **Active Perception**. Its internal "Scouts" and "Cursors" physically move across the text, deciding what to focus on.

### The Brain Modules

1.  **The Thalamus (CEO)**
    *   **Role:** The central router and decision maker.
    *   **Mechanism:** A GRU-based core that receives summary vectors from all other modules, maintains the global state, and issues command vectors back to them. It coordinates the flow of information.

2.  **The Reader (The Eyes)**
    *   **Role:** Scans input text and encodes meaning.
    *   **Mechanism:** Composed of multiple **"Scouts"**.
    *   **Active:** Scouts are not static; they have a "window" of vision (49 chars) and output movement actions (-5, -1, 0, +1, +5) to shift their focus dynamically based on orders from the Thalamus.
    *   **Encoder:** Compresses raw text into "Idea Vectors" (64d).

3.  **The Thinker (The Cortex)**
    *   **Role:** Processing and long-term context retention.
    *   **Mechanism:** A dense memory block that processes "thoughts" (vectors) without direct sensory input. It helps Greg remember context that isn't currently visible to the Reader.

4.  **The Writer (The Hands)**
    *   **Role:** Generates output.
    *   **Mechanism:** Uses **"Cursors"** that can Type, Delete, or Move.
    *   **Output:** Instead of just predicting the next token, it controls a cursor to construct text actively.

## Technical Details
*   **Framework:** PyTorch
*   **Embeddings:** Nibble-based (Splits bytes into two 4-bit parts for 0-255 ASCII support).
*   **Communication:** Modules communicate via dense vectors (128d) routed through the Thalamus.

