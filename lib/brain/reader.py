import torch
import torch.nn as nn
from .config import BrainConfig
from .modules import WindowScoutBase

class ReaderScout(WindowScoutBase):
    """
    Reader Scout: Moves around and reports what it sees.
    """
    def __init__(self):
        super().__init__(
            hidden_size=BrainConfig.SCOUT_HIDDEN_SIZE, 
            window_size=BrainConfig.SCOUT_WINDOW_SIZE,
            input_vector_size=BrainConfig.SCOUT_ORDER_SIZE,
            output_vector_size=BrainConfig.SCOUT_IDEA_SIZE
        )

class Interpreter(nn.Module):
    """
    Reader Interpreter:
    - RNN based.
    - Receives vectors from all scouts.
    - Sends vectors (orders) to all scouts.
    - Sends summary vector to Thalamus.
    - Receives command from Thalamus.
    """
    def __init__(self):
        super().__init__()
        self.num_scouts = BrainConfig.SCOUT_COUNT
        self.scout_idea_size = BrainConfig.SCOUT_IDEA_SIZE
        self.scout_order_size = BrainConfig.SCOUT_ORDER_SIZE
        self.thal_vector_size = BrainConfig.VECTOR_SIZE
        
        # Input: Vectors from all scouts + Command from Thalamus
        input_size = (self.num_scouts * self.scout_idea_size) + self.thal_vector_size
        
        self.rnn = nn.GRU(input_size, BrainConfig.INTERPRETER_HIDDEN_SIZE, batch_first=True)
        
        # Heads
        # 1. Orders for Scouts (One vector per scout)
        self.to_scouts = nn.Linear(BrainConfig.INTERPRETER_HIDDEN_SIZE, self.num_scouts * self.scout_order_size)
        
        # 2. Report to Thalamus
        self.to_thalamus = nn.Linear(BrainConfig.INTERPRETER_HIDDEN_SIZE, self.thal_vector_size)
        
        # 3. Scout Activation (On/Off logits for each scout)
        self.scout_active_logits = nn.Linear(BrainConfig.INTERPRETER_HIDDEN_SIZE, self.num_scouts)

    def forward(self, scout_vectors, thalamus_command, hidden_state=None):
        """
        Args:
            scout_vectors: (Batch, NumScouts, VectorSize)
            thalamus_command: (Batch, VectorSize)
            hidden_state: RNN hidden state
        """
        batch_size = scout_vectors.size(0)
        
        # Flatten scout vectors: (Batch, NumScouts * VectorSize)
        scouts_flat = scout_vectors.view(batch_size, -1)
        
        # Combine inputs
        combined_input = torch.cat([scouts_flat, thalamus_command], dim=1)
        
        # Add sequence dimension for RNN: (Batch, 1, InputSize)
        rnn_input = combined_input.unsqueeze(1)
        
        # RNN Step
        output, next_hidden = self.rnn(rnn_input, hidden_state)
        
        # Flatten output: (Batch, Hidden)
        output = output.squeeze(1)
        
        # Generate outputs
        scout_orders_flat = self.to_scouts(output)
        scout_orders = scout_orders_flat.view(batch_size, self.num_scouts, self.scout_order_size)
        
        to_thalamus = self.to_thalamus(output)
        active_logits = self.scout_active_logits(output)
        
        return scout_orders, to_thalamus, active_logits, next_hidden

class ReaderThing(nn.Module):
    """
    Container for the entire Reader subsystem.
    """
    def __init__(self):
        super().__init__()
        self.scouts = nn.ModuleList([ReaderScout() for _ in range(BrainConfig.SCOUT_COUNT)])
        self.interpreter = Interpreter()
        
    def forward(self, text_windows, thalamus_command, previous_scout_orders, interpreter_hidden):
        """
        Args:
            text_windows: (Batch, NumScouts, WindowSize) - LongTensor indices
            thalamus_command: (Batch, VectorSize)
            previous_scout_orders: (Batch, NumScouts, VectorSize) - From previous step
            interpreter_hidden: RNN hidden state
            
        Returns:
            scout_moves: (Batch, NumScouts, 5) - Logits
            to_thalamus: (Batch, VectorSize)
            new_scout_orders: (Batch, NumScouts, VectorSize)
            scout_active_logits: (Batch, NumScouts)
            new_interpreter_hidden: State
        """
        batch_size = text_windows.size(0)
        scout_outputs_list = []
        scout_moves_list = []
        
        # Run each scout
        for i, scout in enumerate(self.scouts):
            # Extract window and order for this scout
            # windows: (Batch, NumScouts, WinSize) -> (Batch, WinSize)
            win = text_windows[:, i, :]
            order = previous_scout_orders[:, i, :]
            
            vec, moves = scout(win, order)
            scout_outputs_list.append(vec)
            scout_moves_list.append(moves)
            
        # Stack scout outputs: (Batch, NumScouts, VectorSize)
        scout_vectors = torch.stack(scout_outputs_list, dim=1)
        scout_moves = torch.stack(scout_moves_list, dim=1)
        
        # Run Interpreter
        new_orders, to_thal, active_logits, new_hidden = self.interpreter(
            scout_vectors, thalamus_command, interpreter_hidden
        )
        
        return scout_moves, to_thal, new_orders, active_logits, new_hidden
