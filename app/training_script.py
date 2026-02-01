import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys

# Ensure we can import lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.brain.model import GregBrain
from lib.brain.config import BrainConfig
from lib.data_loader import load_text_data, load_fim_data
from lib.coherence import ContradictionDetector

def train_brain(epochs=100, batch_size=16, seq_len=100):
    # --- Enhancement 4: TensorBoard Logs ---
    writer = SummaryWriter('runs/greg_experiment')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Initialize Brain
    brain = GregBrain().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        brain.parameters(), 
        lr=BrainConfig.LEARNING_RATE, 
        weight_decay=BrainConfig.WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # --- Enhancement 1: Coherence Module ---
    coherence_detector = ContradictionDetector()
    
    print("Loading data...")
    # Load standard data
    X_raw, _ = load_text_data(samples=500, sequence_length=seq_len + BrainConfig.SCOUT_WINDOW_SIZE)
    
    # --- Enhancement 2: Fill-In-The-Middle (FIM) Data ---
    print("Generating FIM data...")
    fim_data = load_fim_data(samples=100, max_len=seq_len + BrainConfig.SCOUT_WINDOW_SIZE)
    
    # Combine data
    # Pad FIM data to match X_raw width if needed, or just truncate X_raw
    # For simplicity, we ensure consistent width
    width = X_raw.shape[1]
    combined_data = []
    
    for row in X_raw:
        combined_data.append(row)
        
    for row in fim_data:
        # Pad or crop
        if len(row) > width:
            combined_data.append(row[:width])
        else:
            padded = row + [0] * (width - len(row))
            combined_data.append(padded)
            
    data_tensor = torch.LongTensor(combined_data).to(device)
    print(f"Total Data shape: {data_tensor.shape}")
    
    brain.train()
    print("Starting training loop...")
    
    global_step = 0
    
    # --- Enhancement 3: Topic Prototype for Mood ---
    # We simulate a "Topic" vector. In a real scenario, this comes from the prompt.
    # We'll use a random vector as the "target mood" for this session.
    topic_prototype = torch.randn(256, device=device)
    topic_prototype = F.normalize(topic_prototype, dim=0)
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle data
        perm = torch.randperm(data_tensor.size(0))
        data_tensor = data_tensor[perm]
        
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i : i + batch_size]
            if len(batch) < batch_size: continue
            
            state = brain.init_state(batch_size, device)
            loss_seq = 0
            
            # Tracking for Coherence
            batch_text_history = [""] * batch_size
            
            start_idx = BrainConfig.SCOUT_WINDOW_SIZE
            end_idx = batch.shape[1] - 1
            
            for t in range(start_idx, end_idx):
                window = batch[:, t - BrainConfig.SCOUT_WINDOW_SIZE : t]
                reader_windows = window.unsqueeze(1).expand(-1, BrainConfig.SCOUT_COUNT, -1)
                writer_windows = window.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                
                actions, state = brain(reader_windows, writer_windows, state)
                
                target = batch[:, t]
                writer_logits = actions['cursor_chars'][:, 0, :]
                
                # Base Loss
                ce_loss = criterion(writer_logits, target)
                
                # --- Enhancement 3: Mood Loss ---
                # Cosine similarity between current mood and topic
                current_mood = state['mood'] # (Batch, 256)
                # Normalize
                current_mood_norm = F.normalize(current_mood, dim=1)
                # Similarity with prototype (expanded)
                sim = torch.matmul(current_mood_norm, topic_prototype) # (Batch,)
                
                # Loss: if sim < 0.7, add penalty
                # We want to maximize sim, so minimize (1 - sim)
                mood_loss = torch.mean(torch.relu(0.7 - sim)) 
                
                # --- Enhancement 1: Coherence Penalty ---
                # Decode predicted char for coherence check
                # We use argmax for the check (what the model "wants" to write)
                pred_chars = torch.argmax(writer_logits, dim=1)
                
                coherence_penalty = 0
                if t % 10 == 0: # Check every 10 steps to save compute
                    for b in range(batch_size):
                        char_code = pred_chars[b].item()
                        char = chr(char_code) if 32 <= char_code <= 126 else ""
                        batch_text_history[b] += char
                        
                        # Keep buffer size 500
                        if len(batch_text_history[b]) > 500:
                            batch_text_history[b] = batch_text_history[b][-500:]
                            
                        # Check contradiction against itself (Memory vs New Generation)
                        # We split history into "Old Memory" and "New Gen"
                        if len(batch_text_history[b]) > 50:
                            memory = batch_text_history[b][:-10]
                            gen = batch_text_history[b][-10:]
                            p_contra = coherence_detector.check_contradiction(gen, memory)
                            
                            if p_contra >= 0.5:
                                coherence_penalty += 1.0
                    
                    coherence_penalty = coherence_penalty / batch_size
                
                # Combine Losses
                total_step_loss = ce_loss + (0.5 * mood_loss) + (0.1 * coherence_penalty)
                loss_seq += total_step_loss
                
                # TensorBoard Logging (Sample)
                if global_step % 100 == 0:
                    writer.add_scalar('Loss/Total', total_step_loss.item(), global_step)
                    writer.add_scalar('Loss/Mood', mood_loss.item(), global_step)
                    writer.add_scalar('Metric/MoodSimilarity', torch.mean(sim).item(), global_step)
                    writer.add_scalar('Metric/CoherencePenalty', coherence_penalty, global_step)
                
                global_step += 1
            
            avg_seq_loss = loss_seq / (end_idx - start_idx)
            
            optimizer.zero_grad()
            avg_seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()
            
        print(f"Epoch {epoch} complete. Avg Loss: {avg_seq_loss.item():.4f}")
        
    writer.close()

if __name__ == "__main__":
    train_brain(epochs=5)
