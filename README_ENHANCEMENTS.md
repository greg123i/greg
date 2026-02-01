# Enhancements v2.0 (Story Training Loop)

We have upgraded the training loop with three key modules to improve narrative consistency and control.

## 1. Coherence-Penalty Module
*   **Goal:** Prevent contradictions (e.g., "Greg is in the kitchen" -> "Greg is in the garden").
*   **Mechanism:** A rule-based `ContradictionDetector` scans the last 500 characters of generated text.
*   **Logic:** If `p_contra >= 0.5`, we add a penalty to the loss function.
*   **File:** `lib/coherence.py`

## 2. Fill-In-The-Middle (FIM) Training
*   **Goal:** Teach Greg to connect plot points.
*   **Mechanism:** Data is split into `<|prefix|>...<|suffix|>...` and the model generates the `<|middle|>`.
*   **File:** `lib/data_loader.py`

## 3. Global-Mood GRU (Thalamus)
*   **Goal:** Maintain a consistent emotional tone or topic.
*   **Mechanism:** A parallel GRU in the Thalamus tracks the "Mood Vector" (256d).
*   **Loss:** We enforce that the Mood Vector stays close (Cosine Similarity >= 0.7) to a global "Topic Prototype".
*   **File:** `lib/brain/thalamus.py`

## Usage
Run the training script:
```bash
python app/training_script.py
```

Monitor with TensorBoard:
```bash
tensorboard --logdir runs
```
