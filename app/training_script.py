import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Ensure we can import lib
# Current: greg/stuff/machine_learning/ai_model/training_script.py
# Target: greg/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.brain.model import GregBrain
from lib.brain.config import BrainConfig
from lib.data_loader import load_text_data

def train_brain(epochs=100, batch_size=16, seq_len=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Initialize Brain
    brain = GregBrain().to(device)
    
    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        brain.parameters(), 
        lr=BrainConfig.LEARNING_RATE, 
        weight_decay=BrainConfig.WEIGHT_DECAY
    )
    
    # Loss for Writer (predicting next char)
    criterion = nn.CrossEntropyLoss()
    
    print("Loading data...")
    # Load raw bytes (integers 0-255)
    # Note: Ensure data_loader is updated to return 0-255 ints
    X_raw, Y_raw = load_text_data(samples=1000, sequence_length=seq_len + BrainConfig.SCOUT_WINDOW_SIZE)
    
    # Convert to Tensor
    data_tensor = torch.LongTensor(X_raw).to(device)
    
    print(f"Data shape: {data_tensor.shape}")
    
    brain.train()
    print("Starting training loop...")
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        # Simple batching
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i : i + batch_size]
            if len(batch) < batch_size: continue
            
            # Initialize Brain State
            state = brain.init_state(batch_size, device)
            
            # Sequence Loop (Teacher Forcing Mode)
            # We simulate the cursors scanning the text linearly
            # t is the "cursor position"
            loss_seq = 0
            
            # We need a warm-up for the sliding window
            # Start at index = WindowSize
            start_idx = BrainConfig.SCOUT_WINDOW_SIZE
            end_idx = batch.shape[1] - 1 # Leave room for target
            
            for t in range(start_idx, end_idx):
                # Construct Windows for this timestep
                # Reader sees: [t - WindowSize : t]
                # Writer sees: [t - WindowSize : t] (Simulating it is at 't')
                
                # Slice: (Batch, WindowSize)
                window = batch[:, t - BrainConfig.SCOUT_WINDOW_SIZE : t]
                
                # Expand for multiple scouts/cursors (Forced to see same thing for now)
                # (Batch, NumScouts, WindowSize)
                reader_windows = window.unsqueeze(1).expand(-1, BrainConfig.SCOUT_COUNT, -1)
                writer_windows = window.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                
                actions, state = brain(reader_windows, writer_windows, state)
                
                # Calculate Loss
                # Target: The actual next character at 't'
                target = batch[:, t] # (Batch,)
                
                # Writer Output: We focus on Cursor 0 for learning to write
                # (Batch, NumCursors, Vocab) -> (Batch, Vocab)
                writer_logits = actions['cursor_chars'][:, 0, :] 
                
                loss = criterion(writer_logits, target)
                loss_seq += loss
            
            avg_seq_loss = loss_seq / (end_idx - start_idx)
            
            optimizer.zero_grad()
            avg_seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()
            
            total_loss += avg_seq_loss.item()
            
            if i % (batch_size * 10) == 0:
                 print(f"Epoch {epoch+1} | Batch {i//batch_size} | Loss: {avg_seq_loss.item():.4f}")
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / (len(data_tensor)//batch_size):.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(brain.state_dict(), f"brain_ckpt_{epoch+1}.pt")

if __name__ == "__main__":
    train_brain()