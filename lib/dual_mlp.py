import numpy as np
import os
import sys

# Attempt to locate CUDA manually if it exists but isn't in PATH
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)
    # Also add to PATH for good measure
    os.environ['PATH'] += os.pathsep + cuda_path

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp  # pyright: ignore[reportMissingImports]
    # Test CuPy
    cp.array([1])
    # Test random module (requires curand DLLs)
    cp.random.randn(1)
    xp = cp
    print("DualMLP: Using CuPy (GPU acceleration)")
except (ImportError, Exception) as e:
    xp = np
    print(f"DualMLP: CuPy available but failed ({e}), using NumPy (CPU)")

class ReaderMLP:
    def __init__(self, hidden_size=128):
        # Input: 100 char text + 32 writer feedback
        self.input_size = 100 + 32
        # Output: 32 to writer + 4 move controls (-10, -1, +1, +10)
        self.output_size = 32 + 4
        self.hidden_size = hidden_size
        
        # Weights
        self.w1 = xp.random.randn(self.input_size, hidden_size) * 0.1
        self.b1 = xp.zeros((1, hidden_size))
        self.w2 = xp.random.randn(hidden_size, self.output_size) * 0.1
        self.b2 = xp.zeros((1, self.output_size))
        
    def forward(self, text_input, writer_feedback):
        # text_input: (batch, 100)
        # writer_feedback: (batch, 32)
        self.last_input = xp.hstack([text_input, writer_feedback])
        
        self.h = xp.tanh(xp.dot(self.last_input, self.w1) + self.b1)
        self.out = xp.dot(self.h, self.w2) + self.b2
        
        # Split output
        # First 32: Signal to Writer (Tanh to keep it stable)
        # Last 4: Move Logits (Linear/Softmax logic handled by logic)
        self.signal_out = xp.tanh(self.out[:, :32])
        self.move_logits = self.out[:, 32:]
        
        return self.signal_out, self.move_logits

    def forward_batch(self, text_input, writer_feedback, w1, b1, w2, b2):
        # Batch forward for population
        # text_input: (Pop, 100)
        # writer_feedback: (Pop, 32)
        
        last_input = xp.hstack([text_input, writer_feedback])
        pop_size = last_input.shape[0]
        
        # Robust Reshapes for Weights (Handle Broadcast Artifacts)
        # w1: (Pop, In, Hidden)
        if w1.shape != (pop_size, self.input_size, self.hidden_size):
            try:
                w1 = w1.reshape(pop_size, self.input_size, self.hidden_size)
            except ValueError:
                print(f"DEBUG: Reader w1 shape {w1.shape} incompatible with ({pop_size}, {self.input_size}, {self.hidden_size})")
                raise

        # b1: (Pop, Hidden)
        if b1.size == pop_size * self.hidden_size:
             b1 = b1.reshape(pop_size, self.hidden_size)
        
        # w2: (Pop, Hidden, Out)
        if w2.shape != (pop_size, self.hidden_size, self.output_size):
            try:
                w2 = w2.reshape(pop_size, self.hidden_size, self.output_size)
            except ValueError:
                 print(f"DEBUG: Reader w2 shape {w2.shape} incompatible with ({pop_size}, {self.hidden_size}, {self.output_size})")
                 raise

        # b2: (Pop, Out)
        if b2.size == pop_size * self.output_size:
             b2 = b2.reshape(pop_size, self.output_size)
        
        try:
            # (Pop, 1, 132) @ (Pop, 132, Hidden) -> (Pop, 1, Hidden)
            h = xp.tanh(xp.matmul(last_input[:, None, :], w1) + b1[:, None, :])
            
            # (Pop, 1, Hidden) @ (Pop, Hidden, Out) -> (Pop, 1, Out)
            out = xp.matmul(h, w2) + b2[:, None, :]
            out = out[:, 0, :] # Squeeze
        except ValueError as e:
            print(f"CRASH DEBUG: Reader Matmul Failed. Input={last_input.shape}, w1={w1.shape}, b1={b1.shape}")
            raise e
        
        signal_out = xp.tanh(out[:, :32])
        move_logits = out[:, 32:]
        return signal_out, move_logits

    def mutate(self, sigma=0.01):
        # Apply random noise to weights
        self.w1 += xp.random.randn(*self.w1.shape) * sigma
        self.b1 += xp.random.randn(*self.b1.shape) * sigma
        self.w2 += xp.random.randn(*self.w2.shape) * sigma
        self.b2 += xp.random.randn(*self.b2.shape) * sigma

    def copy_from(self, other):
        # Deep copy weights from another instance
        self.w1 = xp.copy(other.w1)
        self.b1 = xp.copy(other.b1)
        self.w2 = xp.copy(other.w2)
        self.b2 = xp.copy(other.b2)

class WriterMLP:
    def __init__(self, hidden_size=256):
        # Input: 32 from reader + 256 memory
        self.input_size = 32 + 256
        # Output: 10 chars + 256 new memory + 32 feedback + 1 halt
        self.output_size = 10 + 256 + 32 + 1
        self.hidden_size = hidden_size
        
        # Weights
        self.w1 = xp.random.randn(self.input_size, hidden_size) * 0.1
        self.b1 = xp.zeros((1, hidden_size))
        self.w2 = xp.random.randn(hidden_size, self.output_size) * 0.1
        self.b2 = xp.zeros((1, self.output_size))

    def mutate(self, sigma=0.01):
        # Apply random noise to weights
        self.w1 += xp.random.randn(*self.w1.shape) * sigma
        self.b1 += xp.random.randn(*self.b1.shape) * sigma
        self.w2 += xp.random.randn(*self.w2.shape) * sigma
        self.b2 += xp.random.randn(*self.b2.shape) * sigma

    def copy_from(self, other):
        # Deep copy weights from another instance
        self.w1 = xp.copy(other.w1)
        self.b1 = xp.copy(other.b1)
        self.w2 = xp.copy(other.w2)
        self.b2 = xp.copy(other.b2)
        
    def forward(self, reader_signal, memory_state):
        self.last_input = xp.hstack([reader_signal, memory_state])
        
        self.h = xp.tanh(xp.dot(self.last_input, self.w1) + self.b1)
        self.out = xp.dot(self.h, self.w2) + self.b2
        
        # Split outputs
        # 0-10: Text (Sigmoid for 0-1 normalization)
        # 10-266: New Memory (Tanh)
        # 266-298: Feedback to Reader (Tanh)
        # 298-299: Halt (Sigmoid)
        
        self.text_out = 1.0 / (1.0 + xp.exp(-self.out[:, :10]))
        self.mem_out = xp.tanh(self.out[:, 10:266])
        self.feedback_out = xp.tanh(self.out[:, 266:298])
        self.halt_out = 1.0 / (1.0 + xp.exp(-self.out[:, 298:]))
        
        return self.text_out, self.mem_out, self.feedback_out, self.halt_out

    def forward_batch(self, reader_signal, memory_state, w1, b1, w2, b2):
        # Batch forward for population
        # Ensure inputs are 2D
        if reader_signal.ndim > 2: reader_signal = reader_signal.reshape(reader_signal.shape[0], -1)
        if memory_state.ndim > 2: memory_state = memory_state.reshape(memory_state.shape[0], -1)
        
        last_input = xp.hstack([reader_signal, memory_state])
        pop_size = last_input.shape[0]

        # Robust Reshapes
        # w1
        if w1.shape != (pop_size, self.input_size, self.hidden_size):
             try:
                 w1 = w1.reshape(pop_size, self.input_size, self.hidden_size)
             except ValueError:
                 print(f"DEBUG: Writer w1 shape {w1.shape} incompatible with ({pop_size}, {self.input_size}, {self.hidden_size})")
                 raise

        # b1
        if b1.size == pop_size * self.hidden_size:
             b1 = b1.reshape(pop_size, self.hidden_size)

        # w2
        if w2.shape != (pop_size, self.hidden_size, self.output_size):
             try:
                 w2 = w2.reshape(pop_size, self.hidden_size, self.output_size)
             except ValueError:
                 print(f"DEBUG: Writer w2 shape {w2.shape} incompatible with ({pop_size}, {self.hidden_size}, {self.output_size})")
                 raise
        
        # b2
        if b2.size == pop_size * self.output_size:
             b2 = b2.reshape(pop_size, self.output_size)

        try:
            # (Pop, 1, In) @ (Pop, In, Hidden) -> (Pop, 1, Hidden)
            h = xp.tanh(xp.matmul(last_input[:, None, :], w1) + b1[:, None, :])
            out = xp.matmul(h, w2) + b2[:, None, :]
            out = out[:, 0, :]
        except ValueError as e:
            print(f"CRASH DEBUG: Writer Matmul Failed. Input={last_input.shape}, w1={w1.shape}, b1={b1.shape}")
            raise e
        
        text_out = 1.0 / (1.0 + xp.exp(-out[:, :10]))
        mem_out = xp.tanh(out[:, 10:266])
        feedback_out = xp.tanh(out[:, 266:298])
        halt_out = 1.0 / (1.0 + xp.exp(-out[:, 298:]))
        
        return text_out, mem_out, feedback_out, halt_out

    def backward(self, d_text, d_halt, learning_rate=0.001):
        # Gradients for outputs
        d_out = xp.zeros_like(self.out)
        
        # Text: Sigmoid derivative -> d_out = d_text * text * (1 - text)
        d_out[:, :10] = d_text * self.text_out * (1 - self.text_out)
        
        # Memory: We don't have explicit target for memory, so we assume 0 gradient?
        # Actually, in BPTT we would get gradient from NEXT step.
        # Here we ignore it for now (stateless training w.r.t memory target).
        
        # Feedback: Tanh derivative. No direct target, but if Reader backprops?
        # Reader doesn't provide gradient TO feedback (Reader consumes feedback).
        # So we ignore d_feedback unless we have it.
        
        # Halt: Sigmoid derivative
        d_out[:, 298:] = d_halt * self.halt_out * (1 - self.halt_out)
        
        # Backprop through w2
        d_h = xp.dot(d_out, self.w2.T) * (1 - self.h ** 2)
        
        d_w2 = xp.dot(self.h.T, d_out)
        d_b2 = xp.sum(d_out, axis=0, keepdims=True)
        
        d_w1 = xp.dot(self.last_input.T, d_h)
        d_b1 = xp.sum(d_h, axis=0, keepdims=True)
        
        # Update weights
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2
        
        # Return gradient for inputs
        d_input = xp.dot(d_h, self.w1.T)
        d_reader_signal = d_input[:, :32]
        return d_reader_signal

class DualModel:
    def __init__(self):
        self.reader = ReaderMLP()
        self.writer = WriterMLP()
        self.xp = xp
        
        # State
        self.memory = xp.zeros((1, 256))
        self.writer_feedback = xp.zeros((1, 32))
        self.window_idx = 0
        
        # Hyperparameters
        self.learning_rate = 0.001
        
    def get_optimal_population_size(self):
        # Allow user override via env var
        if "DUAL_MLP_POP" in os.environ:
             return int(os.environ["DUAL_MLP_POP"])

        if self.xp == np:
            return 10  # CPU is slow
        try:
            # Check free VRAM
            mem_info = self.xp.cuda.Device(0).mem_info
            free_mem = mem_info[0] # bytes
            
            # Estimate memory per model instance
            # Weights: ~700KB
            # Forward pass intermediates + Workspace: High variance
            # Let's be much more conservative: 20MB per instance
            per_instance = 20 * 1024 * 1024
            
            # Use 50% of free memory (Safety margin for workspaces)
            available = free_mem * 0.5
            count = int(available / per_instance)
            
            # Cap at 5,000 to be safe
            count = min(count, 5000)
            
            # Ensure at least 10
            return max(10, count)
        except:
            return 100

    def get_window_text_vals(self, full_text_array, idx):
        # Safely get 100 chars from full text array at idx
        # full_text_array should be a 1D array of ord() values
        length = len(full_text_array)
        start = max(0, min(idx, length - 100))
        end = start + 100
        
        # Padding if text is too short (shouldn't happen with correct data loading)
        chunk = full_text_array[start:end]
        if len(chunk) < 100:
            padding = xp.zeros(100 - len(chunk))
            chunk = xp.concatenate([chunk, padding])
            
        # Normalize to 0-1
        chunk = chunk / 255.0
            
        return chunk.reshape(1, 100)

    def train_round(self, full_text_input, target_text_seq, max_steps=10):
        """
        Executes one 'Round' of training.
        Returns the loss for this round.
        """
        
        total_loss = 0
        steps_taken = 0
        
        # Ensure input is on correct device
        if not isinstance(full_text_input, self.xp.ndarray):
             full_text_input = self.xp.array(full_text_input)
             
        # Ensure target is on correct device
        if not isinstance(target_text_seq, self.xp.ndarray):
             target_text_seq = self.xp.array(target_text_seq)
        
        # Keep initial state to revert if needed? 
        # No, in this "forward only" pass we just calculate loss.
        # Mutation happens outside.
        
        for step in range(max_steps):
            steps_taken += 1
            
            # 1. Reader
            # Get current view
            current_text = self.get_window_text_vals(full_text_input, self.window_idx)
            r_sig, r_move = self.reader.forward(current_text, self.writer_feedback)
            
            # Move Logic (Discrete)
            # r_move is (1, 4) -> [-10, -1, +1, +10]
            move_idx = int(xp.argmax(r_move))
            moves = [-10, -1, 1, 10]
            chosen_move = moves[move_idx]
            
            # Apply move
            self.window_idx += chosen_move
            self.window_idx = max(0, min(self.window_idx, len(full_text_input) - 100))
            
            # 2. Writer
            # Takes Reader Signal + Memory
            w_text, w_mem, w_feedback, w_halt = self.writer.forward(r_sig, self.memory)
            
            # Update State
            self.memory = w_mem
            self.writer_feedback = w_feedback
            
            # 3. Loss Calculation
            halt_prob = float(w_halt[0,0])
            
            # Text Loss (MSE)
            # Normalize target to 0-1
            target_norm = target_text_seq / 255.0
            
            # We compare output text with target text
            diff = w_text - target_norm
            mse = xp.mean(diff ** 2)
            
            # Penalty (0.01 per step)
            penalty = 0.01 * steps_taken
            
            step_loss = mse + penalty
            total_loss += step_loss
            
            # If Halt is high (> 0.5), we stop and take this as final answer
            if halt_prob > 0.5:
                break
        
        return total_loss

    def copy_from(self, other):
        self.reader.copy_from(other.reader)
        self.writer.copy_from(other.writer)
        # We don't copy state (memory/window) as that is per-instance runtime state

    def mutate(self, sigma=0.01):
        self.reader.mutate(sigma)
        self.writer.mutate(sigma)

    def save_progress(self):
        # Save to a default checkpoint file in the project structure
        # c:\Users\QWE\Desktop\greg\stuff\machine_learning\ai_model\dual_model_checkpoint.npz
        # We need to construct absolute path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, '..', 'stuff', 'machine_learning', 'ai_model')
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'dual_model_checkpoint.npz')
        
        self.save(path)
        print(f"Progress saved to {path}")

    def get_window_text_vals_batch(self, full_text_array, indices):
        # indices: (Pop,)
        # full_text_array: (Len,)
        offsets = self.xp.arange(100)
        # (Pop, 100) = indices[:, None] + offsets[None, :]
        gather_indices = indices[:, None] + offsets[None, :]
        
        # Mask for valid
        valid_mask = gather_indices < len(full_text_array)
        
        # Safe gather
        safe_indices = self.xp.clip(gather_indices, 0, len(full_text_array) - 1)
        chunk = full_text_array[safe_indices]
        
        # Null padding (0) where invalid
        chunk = chunk * valid_mask
        
        # Normalize
        return chunk / 255.0

    def train_loop(self, training_data, epochs=10, rounds_per_epoch=10, callback=None):
        # Clean memory before starting
        if self.xp != np:
            self.xp.get_default_memory_pool().free_all_blocks()
            
        # Determine population size
        pop_size = self.get_optimal_population_size()
        print(f"Training with Population Size: {pop_size}")
        
        # Ensure pop_size is valid
        if pop_size < 1:
            print("Error: Population size calculation failed (size < 1). Defaulting to 10.")
            pop_size = 10
        
        try:
            # Helper to broadcast weights
            def init_pop_tensor(w):
                # Ensure input is array
                if not isinstance(w, self.xp.ndarray):
                     w = self.xp.array(w)
                return self.xp.tile(w[None, ...], (pop_size, 1, 1))
                
            params = {
                'rw1': init_pop_tensor(self.reader.w1),
                'rb1': init_pop_tensor(self.reader.b1),
                'rw2': init_pop_tensor(self.reader.w2),
                'rb2': init_pop_tensor(self.reader.b2),
                'ww1': init_pop_tensor(self.writer.w1),
                'wb1': init_pop_tensor(self.writer.b1),
                'ww2': init_pop_tensor(self.writer.w2),
                'wb2': init_pop_tensor(self.writer.b2),
            }
            
            # Initial mutation
            sigma = 0.02
            for key in params:
                # Skip mutating the first one (keep original)
                if pop_size > 1:
                    params[key][1:] += self.xp.random.randn(*params[key][1:].shape) * sigma
                
            best_loss = float('inf')
            
            for e in range(epochs):
                # 1. Pick Data
                idx = np.random.randint(0, len(training_data))
                full_text = training_data[idx]
                if not isinstance(full_text, self.xp.ndarray):
                    full_text = self.xp.array(full_text, dtype=self.xp.float32)
                if len(full_text) < 200: continue
                
                # 2. Reset State
                pop_memory = self.xp.zeros((pop_size, 256), dtype=self.xp.float32)
                pop_feedback = self.xp.zeros((pop_size, 32), dtype=self.xp.float32)
                
                start_idx = np.random.randint(0, len(full_text) - 150)
                pop_window_idx = self.xp.full((pop_size,), start_idx, dtype=self.xp.int32)
                
                # 3. Evaluate
                pop_losses = self.xp.zeros(pop_size, dtype=self.xp.float32)
                
                for r in range(rounds_per_epoch):
                    # Inputs
                    chunk = self.get_window_text_vals_batch(full_text, pop_window_idx)
                    
                    # Reader
                    r_sig, r_move = self.reader.forward_batch(
                        chunk, pop_feedback, 
                        params['rw1'], params['rb1'], params['rw2'], params['rb2']
                    )
                    
                    # Move
                    # Safety check for empty arrays
                    if r_move.shape[0] == 0:
                        raise ValueError(f"Batch size became zero! pop_size={pop_size}")

                    try:
                        move_indices = self.xp.argmax(r_move, axis=1)
                    except ValueError as e:
                        print(f"CRASH DEBUG: pop_size={pop_size}, r_move.shape={r_move.shape}, r_sig.shape={r_sig.shape}")
                        # Force recovery
                        move_indices = self.xp.zeros((r_move.shape[0],), dtype=self.xp.int32)
                        if r_move.shape[0] == 0: raise e # Re-raise if truly empty

                    # Ensure r_sig is 2D before passing to Writer
                    # It might be 3D if squeeze failed or wasn't enough?
                    if r_sig.ndim > 2:
                        r_sig = r_sig.reshape(r_sig.shape[0], -1)

                    moves_lookup = self.xp.array([-10, -1, 1, 10], dtype=self.xp.int32)
                    pop_window_idx += moves_lookup[move_indices]
                    # Ensure we don't go past the end such that we have no target data
                    # We need 100 chars for input AND 10 chars for target
                    # So max index is len - 110
                    pop_window_idx = self.xp.clip(pop_window_idx, 0, len(full_text) - 110)
                    
                    # Writer
                    w_text, w_mem, w_feedback, w_halt = self.writer.forward_batch(
                        r_sig, pop_memory,
                        params['ww1'], params['wb1'], params['ww2'], params['wb2']
                    )
                    
                    # Update
                    pop_memory = w_mem
                    pop_feedback = w_feedback
                    
                    # Loss (Target: window + 100)
                    # We need to predict next 10 chars
                    # If window moved, target is relative to NEW window or OLD window?
                    # Originally: target_idx = window + 100.
                    target_indices = pop_window_idx + 100
                    target_chunk = self.get_window_text_vals_batch(full_text, target_indices)[:, :10]
                    
                    diff = w_text - target_chunk
                    mse = self.xp.mean(diff ** 2, axis=1)
                    pop_losses += mse
                    
                pop_losses /= rounds_per_epoch
                
                # 4. Selection
                sorted_indices = self.xp.argsort(pop_losses)
                
                if pop_size <= 999:
                    top_pct = 0.10
                elif pop_size < 10000:
                    top_pct = 0.01
                else:
                    top_pct = 0.001
                
                top_n = max(1, int(pop_size * top_pct))
                elites = sorted_indices[:top_n]
                
                best_idx = elites[0]
                current_best_loss = float(pop_losses[best_idx])
                
                # Dynamic Mutation Rate (Adaptive)
                # If we are improving, we can be more precise (smaller sigma)
                # If we are stuck, we need more chaos (larger sigma)
                if current_best_loss < best_loss:
                    best_loss = current_best_loss
                    print(f"Epoch {e}: Best Loss {best_loss:.6f} (New Record)")
                    
                    # Copy Best to Self
                    self.reader.w1 = params['rw1'][best_idx].copy()
                    self.reader.b1 = params['rb1'][best_idx].copy()
                    self.reader.w2 = params['rw2'][best_idx].copy()
                    self.reader.b2 = params['rb2'][best_idx].copy()
                    self.writer.w1 = params['ww1'][best_idx].copy()
                    self.writer.b1 = params['wb1'][best_idx].copy()
                    self.writer.w2 = params['ww2'][best_idx].copy()
                    self.writer.b2 = params['wb2'][best_idx].copy()
                    
                    if e % 5 == 0:
                        self.save_progress()
                        
                    # Decrease mutation (exploit)
                    sigma = max(0.001, sigma * 0.95)
                else:
                    if e % 10 == 0:
                         print(f"Epoch {e}: Best Loss {current_best_loss:.6f} (Stagnant, Global: {best_loss:.6f})")
                    # Increase mutation (explore)
                    sigma = min(0.2, sigma * 1.05)

                # 5. Breed
                new_params = {}
                needed = pop_size - top_n
                
                # Generate indices for parents
                if needed > 0:
                    parent_indices = self.xp.random.randint(0, top_n, needed)
                    # Map back to original population indices
                    real_parent_indices = elites[parent_indices]
                    
                    for key in params:
                        elite_weights = params[key][elites]
                        children = params[key][real_parent_indices].copy()
                        
                        # Mutate children
                        # "Mutants in each epoch"
                        noise = self.xp.random.randn(*children.shape) * sigma
                        children += noise
                        
                        new_params[key] = self.xp.vstack([elite_weights, children])
                else:
                    # If top_n == pop_size (rare, small pop), just keep them
                    for key in params:
                         new_params[key] = params[key][elites]
                    
                params = new_params
            
            self.save_progress()
            
            if callback:
                callback(e, best_loss)
            
        except MemoryError as e:
            print(f"Training Failed: Out of Memory ({e}).")
            print("Try reducing population size or restarting.")
        except Exception as e:
            if "OutOfMemory" in str(e):
                print(f"Training Failed: GPU Out of Memory.")
                print("The population size was too large for your VRAM.")
            else:
                print(f"Training Error: {e}")
                import traceback
                traceback.print_exc()


    def save(self, path):
        # Save weights
        if xp == np:
            np.savez(path, 
                     rw1=self.reader.w1, rb1=self.reader.b1, rw2=self.reader.w2, rb2=self.reader.b2,
                     ww1=self.writer.w1, wb1=self.writer.b1, ww2=self.writer.w2, wb2=self.writer.b2)
        else:
             np.savez(path, 
                     rw1=cp.asnumpy(self.reader.w1), rb1=cp.asnumpy(self.reader.b1), 
                     rw2=cp.asnumpy(self.reader.w2), rb2=cp.asnumpy(self.reader.b2),
                     ww1=cp.asnumpy(self.writer.w1), wb1=cp.asnumpy(self.writer.b1), 
                     ww2=cp.asnumpy(self.writer.w2), wb2=cp.asnumpy(self.writer.b2))

    def load(self, path):
        try:
            d = np.load(path)
            self.reader.w1 = xp.array(d['rw1'])
            self.reader.b1 = xp.array(d['rb1'])
            self.reader.w2 = xp.array(d['rw2'])
            self.reader.b2 = xp.array(d['rb2'])
            self.writer.w1 = xp.array(d['ww1'])
            self.writer.b1 = xp.array(d['wb1'])
            self.writer.w2 = xp.array(d['ww2'])
            self.writer.b2 = xp.array(d['wb2'])
            return True
        except:
            return False
