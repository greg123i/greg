import torch
import numpy as np
import threading
import random
import os
import sys
import time

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports
from lib.brain.model import GregBrain
from lib.brain.config import BrainConfig
from lib.data_loader import load_raw_text_data, load_text_data
from lib.data_tools import download_tinystories_subset, generate_ollama_data, crawl_and_scrape

from lib.data_loader import load_raw_byte_data, FileAnalyzer
from lib.debug_utils import log_debug, log_error
from lib.diagnostics import DiagnosticsManager

class ModelInterface:
    def load_checkpoint(self):
        """Exposed method to load checkpoint logic manually if needed"""
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}...")
            try:
                # Load with weights_only=False since we trust our own checkpoints
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    try:
                        self.brain.load_state_dict(checkpoint['model_state_dict'])
                        print("Checkpoint loaded successfully.")
                        self.status_message = "Loaded previous training progress."
                    except RuntimeError as e:
                        print(f"Size mismatch in checkpoint: {e}")
                        print("Archiving incompatible checkpoint and starting fresh.")
                        
                        # Archive
                        timestamp = int(time.time())
                        archive_path = f"{self.checkpoint_path}.{timestamp}.bak"
                        try:
                            os.rename(self.checkpoint_path, archive_path)
                            print(f"Archived to {archive_path}")
                        except OSError as ren_err:
                            print(f"Could not archive checkpoint: {ren_err}")
                        
                        # Initialize fresh
                        self.status_message = "Architecture changed. Started fresh."
                else:
                    print("Checkpoint format invalid. Starting fresh.")
                    self.status_message = "Invalid checkpoint. Started fresh."
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                self.status_message = f"Load failed ({type(e).__name__}). Started fresh."
        else:
            self.status_message = "Fresh GregBrain initialized."

    def __init__(self):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"ModelInterface using device: {self.device}")
        
        # Model initialization
        try:
            self.brain = GregBrain().to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA Out of Memory during initialization. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.brain = GregBrain().to(self.device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
        
        self.brain_state = None # Holds the persistent state (hidden states)
        
        # Checkpoint Management
        self.checkpoints_dir = os.path.join(project_root, 'checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        self.checkpoint_path = os.path.join(self.checkpoints_dir, 'greg_brain.pth')
        self.elite_path = os.path.join(self.checkpoints_dir, 'greg_brain_elite.pth')
        self.history = [] # List of (epoch, loss, state_dict)
        
        self.load_checkpoint()
        
        self.is_training = False
        self.stop_requested = False # Flag to end training early
        self.training_thread = None
        self.prediction_result = ""
        self.training_sample = ""
        self.diag = {}
        self.loss_history = [] # For graphing
        self.show_loss_graph = False
        
        # Curriculum Learning
        self.difficulty_file = os.path.join(project_root, 'difficulty_scores.json')
        self.difficulty_scores = self.load_difficulty()
        
        # Data
        self.data_directory = os.path.join(project_root, 'app', 'data')
        self.training_inputs = None # Loaded on demand or training start
        self.analyzer = FileAnalyzer(self.data_directory)
        
        # Diagnostics
        self.diag_manager = DiagnosticsManager(os.path.join(project_root, 'diagnostics'))
    
    def load_difficulty(self):
        if os.path.exists(self.difficulty_file):
            try:
                import json
                with open(self.difficulty_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_difficulty(self):
        try:
            import json
            with open(self.difficulty_file, 'w') as f:
                json.dump(self.difficulty_scores, f)
        except:
            pass

    def save_checkpoint(self, path=None):
        target = path or self.checkpoint_path
        print(f"Saving checkpoint to {target}...")
        try:
            torch.save({
                'model_state_dict': self.brain.state_dict(),
                'difficulty': self.difficulty_scores,
                'loss_history': self.loss_history
            }, target)
            print("Checkpoint saved.")
        except Exception as e:
            print(f"Save failed: {e}")

    def monte_carlo_loss(self, num_samples=50, seq_len=50):
        """
        Estimates the average loss across the entire dataset using Monte Carlo sampling.
        """
        if not self.analyzer.file_vectors: return 0
        
        self.brain.eval()
        total_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        # Pick a random cluster and load it
        cluster_ids = list(self.analyzer.clusters.keys())
        if not cluster_ids: return 0
        cid = random.choice(cluster_ids)
        data = self.get_epoch_bytes_for_cluster(cid)
        if data is None or len(data) < seq_len + 10: return 0
        
        data = data.to(self.device) # Move to device for indexing if needed, or keep CPU
        
        with torch.no_grad():
            for _ in range(num_samples):
                if len(data) <= seq_len + 1: break
                start_idx = random.randint(0, len(data) - seq_len - 1)
                seq = data[start_idx : start_idx + seq_len + 1].to(self.device)
                inputs = seq[:-1].unsqueeze(0)
                targets = seq[1:].unsqueeze(0)
                
                state = self.brain.init_state(1, self.device)
                loss_seq = 0
                for t in range(BrainConfig.SCOUT_WINDOW_SIZE, seq_len):
                    window = inputs[:, t - BrainConfig.SCOUT_WINDOW_SIZE : t]
                    r_win = window.unsqueeze(1).expand(-1, BrainConfig.SCOUT_COUNT, -1)
                    w_win = window.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                    actions, state = self.brain(r_win, w_win, state)
                    logits = actions['cursor_chars'][:, 0, :]
                    loss_seq += criterion(logits, targets[:, t])
                
                total_loss += (loss_seq / (seq_len - BrainConfig.SCOUT_WINDOW_SIZE)).item()
        
        return total_loss / max(1, num_samples)
    
    def get_epoch_bytes_for_cluster(self, cid):
        files = self.analyzer.clusters.get(cid, [])
        if not files: return None
        
        # Limit total bytes loaded to avoid RAM spike
        MAX_BYTES = 50 * 1024 * 1024 # 50 MB
        
        buf = bytearray()
        # Shuffle files to get variety
        files_shuffled = list(files)
        random.shuffle(files_shuffled)
        
        for fp in files_shuffled:
            if len(buf) >= MAX_BYTES: break
            try:
                # If file is huge, read random chunk
                fsize = os.path.getsize(fp)
                if fsize > MAX_BYTES:
                    with open(fp, 'rb') as f:
                        start = random.randint(0, fsize - MAX_BYTES)
                        f.seek(start)
                        b = f.read(MAX_BYTES)
                        b = b.replace(b"<|endoftext|>", b"").replace(b"<|endoftext", b"")
                        buf.extend(b)
                else:
                    with open(fp, 'rb') as f:
                        b = f.read().replace(b"<|endoftext|>", b"").replace(b"<|endoftext", b"")
                        buf.extend(b)
                buf.append(32) # Space separator
            except Exception:
                continue
                
        return torch.tensor(list(buf), dtype=torch.long) if buf else None
    
    def get_diagnostics(self):
        metrics = self.diag_manager.get_latest()
        lines = []
        lines.append(f"Device: {self.device}")
        lines.append(f"ScoutWin: {BrainConfig.SCOUT_WINDOW_SIZE} Scouts: {BrainConfig.SCOUT_COUNT} Cursors: {BrainConfig.CURSOR_COUNT}")
        
        if 'selected_file' in self.diag:
            lines.append(f"File: {self.diag.get('selected_file')}")
            
        lines.append("-" * 20)
        lines.append(f"Epoch: {metrics.get('epoch', 0)} | Step: {metrics.get('step', 0)}")
        lines.append(f"Loss: {metrics.get('loss', 0.0):.6f}")
        lines.append(f"LR: {metrics.get('learning_rate', 0.0):.2e}")
        lines.append(f"Grad Norm: {metrics.get('grad_norm', 0.0):.4f}")
        
        ppx = metrics.get('perplexity', 0.0)
        ppx_str = f"{ppx:.2f}" if ppx < 10000 else "Inf"
        lines.append(f"Perplexity: {ppx_str}")
        
        lines.append(f"Speed: {metrics.get('tokens_per_sec', 0.0):.1f} tok/s")
        
        if metrics.get('anomalies'):
            lines.append("--- ANOMALIES DETECTED ---")
            for a in metrics['anomalies']:
                lines.append(f"[!] {a}")
                
        return lines

    def export_diagnostics(self):
        return self.diag_manager.save_snapshot()

    def start_training(self, epochs=2, persistent_memory=True, batch_size=16, seq_len=50, elitism_epochs=25, target_loss=1e-5, time_minutes=None):
        if self.is_training:
            return
        
        # Enforce "Only One Active Condition" Rule
        # Priority: Time > Epochs > Target Loss
        # If Time is set (>0), it overrides everything else.
        # If Epochs is set (>0), it overrides Target Loss.
        # If Epochs is 0, it runs until Target Loss is met.
        
        stop_condition_mode = "epochs"
        
        if time_minutes is not None and float(time_minutes) > 0:
            stop_condition_mode = "time"
            epochs = 999999999 # Virtually infinite
            target_loss = -999.0 # Disable loss check
            self.status_message = f"Training Mode: Time Limit ({time_minutes} min)"
            
        elif int(epochs) > 0:
            stop_condition_mode = "epochs"
            target_loss = -999.0 # Disable loss check
            time_minutes = None # Disable time check
            self.status_message = f"Training Mode: Fixed Epochs ({epochs})"
      
        else:
            stop_condition_mode = "loss"
            epochs = 999999999 # Virtually infinite
            time_minutes = None # Disable time check
            self.status_message = f"Training Mode: Target Loss ({target_loss})"

        self.is_training = True
        
        def train_task():
            try:
                # Note: We NO LONGER load all data into self.training_inputs to save RAM.
                # Data is loaded on-the-fly per epoch using get_epoch_bytes_for_cluster.
                
                # Ensure we have files to index
                if not os.path.exists(self.data_directory) or not os.listdir(self.data_directory):
                     print("No data found. Generating sample data...")
                     os.makedirs(self.data_directory, exist_ok=True)
                     sample_path = os.path.join(self.data_directory, "sample_conversation.txt")
                     with open(sample_path, "w", encoding="utf-8") as f:
                         # Generate conversational sample data
                         conversations = [
                             "User: Hello|AI: Hi there! I am Greg.",
                             "User: How are you?|AI: I am doing well, thank you.",
                             "User: What is your name?|AI: My name is Greg.",
                             "User: Write a story.|AI: Once upon a time, there was an AI.",
                             "User: Help me.|AI: I can help you with your tasks."
                         ]
                         # Repeat to create enough volume
                         for _ in range(200):
                             f.write("\n".join(conversations) + "\n")
                
                # Vectorize and cluster (Metadata only, light on RAM)
                self.analyzer.vectorize_files()
                self.analyzer.cluster_files() # Ensure clusters exist for MC loss
                
                if not self.analyzer.file_vectors:
                    self.status_message = "Training Error: No data files found."
                    self.is_training = False
                    return

                # Initialize difficulty for new files and prune old ones
                existing_files = set(self.analyzer.file_vectors.keys())
                
                # Prune missing files from difficulty scores
                for fpath in list(self.difficulty_scores.keys()):
                    if fpath not in existing_files and not os.path.exists(fpath):
                        del self.difficulty_scores[fpath]
                
                # Add new files
                for fpath in existing_files:
                    if fpath not in self.difficulty_scores:
                        self.difficulty_scores[fpath] = 1.0
                
                # Verify we have playable files
                if not self.difficulty_scores:
                     self.status_message = "No valid files to train on."
                     self.is_training = False
                     return
                
                optimizer = torch.optim.AdamW(self.brain.parameters(), lr=BrainConfig.LEARNING_RATE, weight_decay=BrainConfig.WEIGHT_DECAY)
                # Scheduler to handle plateaus
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
                criterion = torch.nn.CrossEntropyLoss()
                
                # Persistent state
                persistent_state = None
                self.stop_requested = False
                
                import time
                start_time = time.time()
                for epoch in range(epochs):
                    if not self.is_training or self.stop_requested: break
                    
                    self.brain.train()
                    total_epoch_loss = 0
                    
                    # Fix: Safe seq_len jitter to avoid empty range
                    # Ensure seq_len is large enough for Scout Window
                    safe_seq_len = max(BrainConfig.SCOUT_WINDOW_SIZE + 10, int(seq_len))
                    actual_seq_len = random.randint(safe_seq_len - 5, safe_seq_len + 5)
                    
                    # --- CURRICULUM SAMPLING ---
                    # Sort files by difficulty
                    sorted_files = sorted(self.difficulty_scores.keys(), key=lambda x: self.difficulty_scores[x], reverse=True)
                    # Take top 5%
                    top_count = max(1, int(len(sorted_files) * 0.05))
                    top_files = sorted_files[:top_count]
                    # Select a random file from top 5%
                    selected_file = random.choice(top_files)
                    
                    # Load data for this file
                    try:
                        with open(selected_file, 'rb') as f:
                            b = f.read().replace(b"<|endoftext|>", b"").replace(b"<|endoftext", b"")
                            epoch_data = torch.tensor(list(b), dtype=torch.long)
                    except Exception as e:
                        print(f"Error loading {selected_file}: {e}")
                        if selected_file in self.difficulty_scores:
                            del self.difficulty_scores[selected_file]
                        continue
                    
                    if len(epoch_data) < actual_seq_len + 10:
                        continue
                        
                    self.diag['selected_file'] = os.path.basename(selected_file)
                    self.diag['epoch'] = epoch + 1
                    self.diag['epochs'] = epochs
                    self.diag['mode'] = stop_condition_mode
                    
                    if persistent_memory:
                        # TBPTT logic...
                        total_tokens = len(epoch_data)
                        chunk_size = total_tokens // batch_size
                        if chunk_size < actual_seq_len + 1: continue
                        
                        stream_starts = [i * chunk_size for i in range(batch_size)]
                        num_steps = (chunk_size - 1) // actual_seq_len
                        
                        if persistent_state is None:
                            persistent_state = self.brain.init_state(batch_size, self.device)
                        else:
                            persistent_state = self.brain.detach_state(persistent_state)

                        for step in range(num_steps):
                            if not self.is_training or self.stop_requested: break
                            
                            batch_indices = []
                            for i in range(batch_size):
                                start = stream_starts[i] + (step * actual_seq_len)
                                end = start + actual_seq_len + 1
                                if end > len(epoch_data): break
                                batch_indices.append(epoch_data[start:end])
                            
                            if len(batch_indices) < batch_size: break
                            
                            batch = torch.stack(batch_indices).to(self.device)
                            inputs, targets = batch[:, :-1], batch[:, 1:]
                            
                            optimizer.zero_grad()
                            state = persistent_state
                            
                            all_windows = inputs.unfold(1, BrainConfig.SCOUT_WINDOW_SIZE, 1)
                            seq_steps = actual_seq_len - BrainConfig.SCOUT_WINDOW_SIZE
                            
                            loss_seq = 0
                            t0 = time.time()
                            for t_step in range(seq_steps):
                                t = t_step + BrainConfig.SCOUT_WINDOW_SIZE
                                
                                scout_windows_list = []
                                # Scout viewing logic (look ahead/behind)
                                left_idx = max(0, t_step - BrainConfig.SCOUT_WINDOW_SIZE)
                                center_idx = t_step
                                right_idx = min(seq_steps - 1, t_step + BrainConfig.SCOUT_WINDOW_SIZE)
                                scout_windows_list.append(all_windows[:, left_idx])
                                scout_windows_list.append(all_windows[:, center_idx])
                                scout_windows_list.append(all_windows[:, right_idx])
                                
                                r_win = torch.stack(scout_windows_list, dim=1)
                                window = all_windows[:, t_step]
                                
                                # Fix: Slice for writer window size
                                w_slice = window[:, -BrainConfig.CURSOR_WINDOW_SIZE:]
                                w_win = w_slice.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                                
                                actions, state = self.brain(r_win, w_win, state)
                                logits = actions['cursor_chars'][:, 0, :]
                                step_loss = criterion(logits, targets[:, t])
                                loss_seq += step_loss
                            
                            avg_loss = loss_seq / seq_steps
                            avg_loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
                            optimizer.step()
                            
                            dt = (time.time() - t0) * 1000.0
                            self.diag['step_time_ms'] = dt
                            tokens_per_sec = (seq_steps * batch_size) / max(1e-6, (dt / 1000.0))
                            self.diag['tokens_per_sec'] = tokens_per_sec

                            # Diagnostics Update
                            self.diag_manager.update({
                                "loss": avg_loss.item(),
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "grad_norm": grad_norm.item(),
                                "step": step,
                                "epoch": epoch + 1,
                                "tokens_per_sec": tokens_per_sec
                            })
                            
                            persistent_state = self.brain.detach_state(state)
                            total_epoch_loss += avg_loss.item()
                            
                            # Track loss for graphing
                            if step % 10 == 0:
                                self.loss_history.append(avg_loss.item())
                                if len(self.loss_history) > 1000: self.loss_history.pop(0)
                                self.status_message = f"Ep {epoch+1} | Loss: {avg_loss.item():.4f}"
                            
                            # Update difficulty: Increase by tiny bit every step
                            self.difficulty_scores[selected_file] += 0.001
                            
                            if time_minutes is not None:
                                if (time.time() - start_time) >= (time_minutes * 60):
                                    self.status_message = "Stopped: time limit reached."
                                    raise StopIteration()
                        
                        # Epoch finished: If loss is low, decrease difficulty
                        avg_epoch_loss = total_epoch_loss / max(1, num_steps)
                        if avg_epoch_loss < 2.0:
                            self.difficulty_scores[selected_file] = max(0.1, self.difficulty_scores[selected_file] - 0.5)
                        
                        scheduler.step(avg_epoch_loss)
                        self.diag['last_loss'] = avg_epoch_loss
                        self.save_difficulty()


                    else:
                        # Old Random Logic (Non-Persistent)
                        steps_per_epoch = 100 
                        
                        for i in range(steps_per_epoch):
                            if not self.is_training or self.stop_requested: break
                            
                            batch_indices = []
                            for _ in range(batch_size):
                                start_idx = random.randint(0, len(epoch_data) - actual_seq_len - 1)
                                batch_indices.append(epoch_data[start_idx : start_idx + actual_seq_len + 1])
                            
                            batch = torch.stack(batch_indices).to(self.device)
                            inputs, targets = batch[:, :-1], batch[:, 1:]
                            state = self.brain.init_state(batch_size, self.device)
                            
                            optimizer.zero_grad()
                            loss_seq = 0
                            
                            all_windows = inputs.unfold(1, BrainConfig.SCOUT_WINDOW_SIZE, 1)
                            seq_steps = actual_seq_len - BrainConfig.SCOUT_WINDOW_SIZE
                            
                            t0 = time.time()
                            for t_step in range(seq_steps):
                                t = t_step + BrainConfig.SCOUT_WINDOW_SIZE
                                
                                scout_windows_list = []
                                left_idx = max(0, t_step - BrainConfig.SCOUT_WINDOW_SIZE)
                                center_idx = t_step
                                right_idx = min(seq_steps - 1, t_step + BrainConfig.SCOUT_WINDOW_SIZE)
                                scout_windows_list.append(all_windows[:, left_idx])
                                scout_windows_list.append(all_windows[:, center_idx])
                                scout_windows_list.append(all_windows[:, right_idx])
                                
                                r_win = torch.stack(scout_windows_list, dim=1)
                                window = all_windows[:, t_step]
                                
                                # Fix: Slice for writer window size
                                w_slice = window[:, -BrainConfig.CURSOR_WINDOW_SIZE:]
                                w_win = w_slice.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                                
                                actions, state = self.brain(r_win, w_win, state)
                                logits = actions['cursor_chars'][:, 0, :]
                                step_loss = criterion(logits, targets[:, t])
                                loss_seq += step_loss
                            
                            avg_loss = loss_seq / seq_steps
                            avg_loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
                            optimizer.step()
                            
                            dt = (time.time() - t0) * 1000.0
                            self.diag['step_time_ms'] = dt
                            tokens_per_sec = (seq_steps * batch_size) / max(1e-6, (dt / 1000.0))
                            self.diag['tokens_per_sec'] = tokens_per_sec

                            # Diagnostics Update
                            self.diag_manager.update({
                                "loss": avg_loss.item(),
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "grad_norm": grad_norm.item(),
                                "step": i,
                                "epoch": epoch + 1,
                                "tokens_per_sec": tokens_per_sec
                            })
                            total_epoch_loss += avg_loss.item()
                            
                            if i % 10 == 0:
                                self.loss_history.append(avg_loss.item())
                                if len(self.loss_history) > 1000: self.loss_history.pop(0)
                                self.status_message = f"Ep {epoch+1} | Loss: {avg_loss.item():.4f}"
                            
                            # Increase difficulty
                            self.difficulty_scores[selected_file] += 0.001

                            if time_minutes is not None:
                                if (time.time() - start_time) >= (time_minutes * 60):
                                    self.status_message = "Stopped: time limit reached."
                                    raise StopIteration()

                        avg_epoch_loss = total_epoch_loss / max(1, steps_per_epoch)
                        if avg_epoch_loss < 2.0:
                            self.difficulty_scores[selected_file] = max(0.1, self.difficulty_scores[selected_file] - 0.5)
                        
                        scheduler.step(avg_epoch_loss)
                        self.diag['last_loss'] = avg_epoch_loss
                        self.save_difficulty()

                    # --- EPOCH PREDICTION ---
                    # Have Greg predict what happens next to some text
                    self.brain.eval()
                    with torch.no_grad():
                        sample_text = "Once upon a time, Greg the AI "
                        input_chars = [ord(c) for c in sample_text]
                        input_tensor = torch.tensor(input_chars, dtype=torch.long).unsqueeze(0).to(self.device)
                        
                        state = self.brain.init_state(1, self.device)
                        prediction = sample_text
                        
                        # Process existing text to build hidden state
                        for t in range(BrainConfig.SCOUT_WINDOW_SIZE, len(input_chars)):
                            window = input_tensor[:, t-BrainConfig.SCOUT_WINDOW_SIZE:t]
                            r_win = window.unsqueeze(1).expand(-1, BrainConfig.SCOUT_COUNT, -1)
                            w_win = window.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                            actions, state = self.brain(r_win, w_win, state)
                        
                        # Generate new text
                        pred_chars = [ord(c) for c in prediction]
                        
                        for _ in range(30):
                            # Construct window from last chars
                            if len(pred_chars) < BrainConfig.SCOUT_WINDOW_SIZE:
                                window_chars = [0] * (BrainConfig.SCOUT_WINDOW_SIZE - len(pred_chars)) + pred_chars
                            else:
                                window_chars = pred_chars[-BrainConfig.SCOUT_WINDOW_SIZE:]
                                
                            last_window = torch.tensor(window_chars, dtype=torch.long).unsqueeze(0).to(self.device)
                            
                            r_win = last_window.unsqueeze(1).expand(-1, BrainConfig.SCOUT_COUNT, -1)
                            w_win = last_window.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                            actions, state = self.brain(r_win, w_win, state)
                            
                            # Cursor 0 Logic
                            # 1. Action Type (0=NoOp, 1=Insert, 2=Overwrite, 3=Delete)
                            act_logits = actions['cursor_actions'][:, 0, :]
                            act_probs = torch.softmax(act_logits, dim=-1)
                            action = torch.multinomial(act_probs, 1).item()
                            
                            # 2. Offset (Where in window to act)
                            off_logits = actions['cursor_offsets'][:, 0, :]
                            off_probs = torch.softmax(off_logits, dim=-1)
                            offset_rel = torch.multinomial(off_probs, 1).item()
                            
                            # Calculate absolute position in pred_chars
                            base_idx = max(0, len(pred_chars) - BrainConfig.SCOUT_WINDOW_SIZE)
                            target_idx = min(len(pred_chars), base_idx + offset_rel)
                            
                            # 3. Character (Top-K Selection)
                            char_logits = actions['cursor_chars'][:, 0, :]
                            top_k = BrainConfig.CURSOR_TOP_K
                            top_probs, top_indices = torch.topk(torch.softmax(char_logits, dim=-1), top_k)
                            idx_in_top = torch.multinomial(top_probs, 1)
                            next_char_idx = top_indices.gather(1, idx_in_top).item()
                            
                            # Apply Action
                            if action == 1: # Insert
                                pred_chars.insert(target_idx, next_char_idx)
                            elif action == 2: # Overwrite
                                if target_idx < len(pred_chars):
                                    pred_chars[target_idx] = next_char_idx
                                else:
                                    pred_chars.append(next_char_idx)
                            elif action == 3: # Delete
                                if target_idx < len(pred_chars):
                                    pred_chars.pop(target_idx)
                            else: # NoOp
                                # For demo purposes, if model does nothing, we force append to show something happening?
                                # No, let's respect the model. But if it stalls, maybe just append.
                                # Let's assume NoOp is valid.
                                pass
                        
                        prediction = "".join([chr(min(255, c)) for c in pred_chars])
                        
                        self.prediction_result = prediction
                        self.diag_manager.update({"sample_output": prediction})
                        print(f"Epoch {epoch+1} Prediction: {prediction}")

                    # End of Epoch Tasks
                    if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                        mc_loss = self.monte_carlo_loss()
                        print(f"Epoch {epoch+1} MC Loss: {mc_loss:.6f}")
                        self.diag['mc_loss'] = mc_loss
                        
                        if mc_loss < target_loss:
                            self.status_message = f"Target loss {target_loss} reached!"
                            self.save_checkpoint()
                            break

                    
                    # Elitism
                    if (epoch + 1) % elitism_epochs == 0:
                        self.save_checkpoint(self.elite_path)
                        if (epoch + 1) % 5 != 0:
                            mc_loss = self.monte_carlo_loss()
                        
                        self.history.append((epoch + 1, mc_loss, self.brain.state_dict()))
                        
                        if len(self.history) >= 3:
                            old_epoch, old_loss, old_state = self.history[-3]
                            if mc_loss > old_loss:
                                print(f"Current loss {mc_loss} > old loss {old_loss}. Reverting!")
                                self.brain.load_state_dict(old_state)
                                # If reverting, maybe reset persistent state too?
                                persistent_state = None 
                                self.status_message = "Reverted to elite!"
                    
                    if (epoch + 1) % 5 == 0:
                        self.save_checkpoint()
                
                # Save at the end of training
                self.save_checkpoint()
                self.status_message = "Training complete!"
            except StopIteration:
                self.save_checkpoint()
                self.status_message = "Training stopped (Time limit)."
            except Exception as e:
                print(f"Training Error: {e}")
                import traceback
                traceback.print_exc()
                self.status_message = f"Error: {e}"
            finally:
                self.is_training = False
            
        self.training_thread = threading.Thread(target=train_task)
        self.training_thread.start()

    def stop_training(self):
        self.stop_requested = True
        self.status_message = "Stop requested..."

    def fetch_training_data(self, target_count=10000):
        if self.is_training:
            return
        
        def fetch_task():
            self.status_message = "Crawling for data (Target: 10k files)..."
            try:
                success = crawl_and_scrape(self.data_directory, target_count=target_count)
                if success:
                    self.status_message = "Data collection complete!"
                else:
                    self.status_message = "Data collection finished."
            except Exception as e:
                self.status_message = f"Crawl Error: {e}"
            
            # Re-analyze after new data
            self.analyzer.vectorize_files()
            
        threading.Thread(target=fetch_task, daemon=True).start()

    def get_random_text(self):
        if self.training_inputs is not None:
             idx = random.randint(0, len(self.training_inputs) - 100)
             chunk = self.training_inputs[idx:idx+100]
             return bytes(chunk).decode('utf-8', errors='replace')
        return "Load data first to see samples."

    def predict_greg(self, prompt_text, target_len=None):
        """
        Robust prediction loop for GregBrain.
        Handles cursor actions (Insert, Overwrite, Delete) with stall protection.
        """
        log_debug(f"predict_greg called with prompt: '{prompt_text}'")
        self.brain.eval()
        with torch.no_grad():
            # 1. Sanitize & Prepare Input
            # Ensure all chars are 0-255. Replace others with '?' (63)
            prompt_bytes = [ord(c) if ord(c) < 256 else 63 for c in prompt_text]
            
            # Pad with spaces if shorter than window
            while len(prompt_bytes) < BrainConfig.SCOUT_WINDOW_SIZE:
                prompt_bytes.insert(0, 32)
            
            log_debug(f"Padded prompt bytes len: {len(prompt_bytes)}")

            # 2. Initialize State
            # We process the prompt to "warm up" the hidden state (teacher forcing)
            curr_seq = torch.tensor(prompt_bytes, dtype=torch.long, device=self.device).unsqueeze(0)
            state = self.brain.init_state(1, self.device)
            
            # Warmup: Run through the prompt
            for t in range(BrainConfig.SCOUT_WINDOW_SIZE, curr_seq.shape[1]):
                # Reader Window
                r_slice = curr_seq[:, t - BrainConfig.SCOUT_WINDOW_SIZE : t]
                r_win = r_slice.unsqueeze(1).expand(-1, BrainConfig.SCOUT_COUNT, -1)
                
                # Writer Window
                w_slice = curr_seq[:, t - BrainConfig.CURSOR_WINDOW_SIZE : t]
                w_win = w_slice.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                
                _, state = self.brain(r_win, w_win, state)
            
            # 3. Generation Loop
            pred_chars = list(prompt_bytes) # Working mutable list
            original_len = len(pred_chars)  # Boundary of user prompt
            
            if target_len is None:
                target_len = random.randint(50, 200) # Shorter default for responsiveness
            
            max_steps = target_len * 3 # Allow some editing overhead
            stall_counter = 0
            
            log_debug(f"Starting generation loop. Target len: {target_len}, Max steps: {max_steps}")

            for step in range(max_steps):
                # Check length limit
                current_len = len(pred_chars)
                generated_count = current_len - original_len
                if generated_count >= target_len:
                    log_debug("Target length reached.")
                    break
                
                # Construct Reader Input Window (Last SCOUT_WINDOW_SIZE chars)
                if current_len < BrainConfig.SCOUT_WINDOW_SIZE:
                    # Pad left with spaces if we deleted too much
                    window_chars = [32] * (BrainConfig.SCOUT_WINDOW_SIZE - current_len) + pred_chars
                else:
                    window_chars = pred_chars[-BrainConfig.SCOUT_WINDOW_SIZE:]
                
                r_window = torch.tensor(window_chars, dtype=torch.long, device=self.device).unsqueeze(0)
                r_win = r_window.unsqueeze(1).expand(-1, BrainConfig.SCOUT_COUNT, -1)
                
                # Construct Writer Input Window (Last CURSOR_WINDOW_SIZE chars)
                if current_len < BrainConfig.CURSOR_WINDOW_SIZE:
                    w_chars = [32] * (BrainConfig.CURSOR_WINDOW_SIZE - current_len) + pred_chars
                else:
                    w_chars = pred_chars[-BrainConfig.CURSOR_WINDOW_SIZE:]
                    
                w_window = torch.tensor(w_chars, dtype=torch.long, device=self.device).unsqueeze(0)
                w_win = w_window.unsqueeze(1).expand(-1, BrainConfig.CURSOR_COUNT, -1)
                
                # Forward Pass
                actions, state = self.brain(r_win, w_win, state)
                
                # --- ACTION LOGIC (Cursor 0) ---
                
                # 1. Action Type: 0=NoOp, 1=Insert, 2=Overwrite, 3=Delete
                act_logits = actions['cursor_actions'][:, 0, :]
                act_probs = torch.softmax(act_logits, dim=-1)
                action = torch.multinomial(act_probs, 1).item()
                
                # 2. Offset (Where relative to window end)
                off_logits = actions['cursor_offsets'][:, 0, :]
                off_probs = torch.softmax(off_logits, dim=-1)
                offset_rel = torch.multinomial(off_probs, 1).item()
                
                # 3. Character Selection
                char_logits = actions['cursor_chars'][:, 0, :]
                top_k = BrainConfig.CURSOR_TOP_K
                top_probs, top_indices = torch.topk(torch.softmax(char_logits, dim=-1), top_k)
                idx_in_top = torch.multinomial(top_probs, 1)
                next_char_idx = top_indices.gather(1, idx_in_top).item()
                
                # --- STALL PROTECTION ---
                # If model chooses NoOp (0) or Delete (3) repeatedly without growing, force Append.
                if action == 0: 
                    stall_counter += 1
                elif action == 3: # Delete
                    stall_counter += 0.5 # Deleting is valid but too much is bad
                else:
                    stall_counter = max(0, stall_counter - 1) # Reset if productive
                
                # Force Overwrite/Append if stalling
                if stall_counter > 5:
                    action = 2 # Force Overwrite/Append
                    offset_rel = BrainConfig.SCOUT_WINDOW_SIZE # Force end of window
                    stall_counter = 0
                    # Ensure we output a visible char if stalling
                    if next_char_idx < 32:
                         next_char_idx = random.choice([32, 46, 63, 33]) # Space, ., ?, !
                
                # --- APPLY ACTION ---
                
                # Calculate absolute target index
                # offset_rel is 0..WindowSize. 
                # 0 means "at the start of window", WindowSize means "at the end (append)"
                # We map this to the end of pred_chars.
                
                # Base is the start of the current window in the full sequence
                base_idx = max(0, len(pred_chars) - BrainConfig.SCOUT_WINDOW_SIZE)
                target_idx = base_idx + offset_rel
                
                # Clamp target_idx
                target_idx = min(len(pred_chars), max(0, target_idx))
                
                # CRITICAL: Prompt Protection
                # Never allow editing before the prompt ends
                if target_idx < original_len:
                    target_idx = len(pred_chars) # Redirect to end (Append)
                    action = 2 # Force Append mode
                
                if action == 1: # Insert
                    pred_chars.insert(target_idx, next_char_idx)
                elif action == 2: # Overwrite / Append
                    if target_idx < len(pred_chars):
                        pred_chars[target_idx] = next_char_idx
                    else:
                        pred_chars.append(next_char_idx)
                elif action == 3: # Delete
                    if target_idx < len(pred_chars):
                        pred_chars.pop(target_idx)
                # else NoOp: do nothing
                
                # --- STOP TOKEN CHECK ---
                # Check for "User:" in the *newly generated* portion only
                # We strictly look after original_len to avoid matching the prompt itself
                
                generated_chars = pred_chars[original_len:]
                generated_str = "".join([chr(c) if c < 256 else '?' for c in generated_chars])
                
                if "User:" in generated_str:
                    log_debug("Stop token 'User:' found. Truncating.")
                    # Found stop token!
                    clean_gen = generated_str.split("User:")[0]
                    
                    # Reconstruct pred_chars to contain [Padding + Prompt] + [Clean Generated]
                    clean_bytes = [ord(c) if ord(c) < 256 else 63 for c in clean_gen]
                    pred_chars = pred_chars[:original_len] + clean_bytes
                    break
            
            # 4. Finalize Output
            final_text = "".join([chr(c) if c < 256 else '?' for c in pred_chars])
            
            # Extract only the generated part (excluding prompt and padding)
            generated_text = final_text[original_len:]
            
            log_debug(f"Generated raw text: '{generated_text}' (len: {len(generated_text)})")

            # Sanitize for GUI
            generated_text = "".join([c for c in generated_text if ord(c) >= 32 or c == '\n'])
            
            if not generated_text:
                log_error("Generated text is empty after sanitization!")
                generated_text = "[Error: Model generated no visible text. See greg_debug.log]"
            
            self.prediction_result = f"Greg: {generated_text}"
            self.status_message = f"Replied ({len(generated_text)} chars)"
            return generated_text

    def generate_chat_response(self, chat_history, user_message):
        # Wrapper for Chat UI
        prompt = f"User: {user_message}|AI:"
        response = self.predict_greg(prompt)
        chat_history.append(f"User: {user_message}")
        chat_history.append(f"AI: {response}")
        return response

    # Legacy / Unused methods required by UI if any?
    # The UI calls:
    # - start_training
    # - fetch_training_data
    # - generate_chat_response
    # - get_random_text (optional)
    # - status_message (property)
    # - training_sample (property)
    
    def predict_dual(self, text):
        # Alias for compatibility if needed
        return self.predict_greg(text)
