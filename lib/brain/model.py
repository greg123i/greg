import torch
import torch.nn as nn
from .config import BrainConfig
from .reader import ReaderThing
from .thinker import ThinkerThing
from .writer import WriterThing
from .thalamus import Thalamus

class GregBrain(nn.Module):
    """
    The complete Brain architecture.
    """
    def __init__(self):
        super().__init__()
        
        self.reader = ReaderThing()
        self.thinker = ThinkerThing()
        self.writer = WriterThing()
        self.thalamus = Thalamus()
        
        # State Initialization helpers
        self.config = BrainConfig

    def init_state(self, batch_size, device):
        """
        Initialize all hidden states and vectors.
        """
        # Vectors
        reader_cmd = torch.zeros(batch_size, BrainConfig.VECTOR_SIZE, device=device)
        thinker_cmd = torch.zeros(batch_size, BrainConfig.VECTOR_SIZE, device=device)
        writer_cmd = torch.zeros(batch_size, BrainConfig.VECTOR_SIZE, device=device)
        
        reader_out = torch.zeros(batch_size, BrainConfig.VECTOR_SIZE, device=device)
        thinker_out = torch.zeros(batch_size, BrainConfig.VECTOR_SIZE, device=device)
        writer_out = torch.zeros(batch_size, BrainConfig.VECTOR_SIZE, device=device)
        
        # Sub-module Vectors
        scout_orders = torch.zeros(batch_size, BrainConfig.SCOUT_COUNT, BrainConfig.SCOUT_ORDER_SIZE, device=device)
        cursor_orders = torch.zeros(batch_size, BrainConfig.CURSOR_COUNT, BrainConfig.VECTOR_SIZE, device=device)
        
        # Hidden States
        interpreter_hidden = torch.zeros(1, batch_size, BrainConfig.INTERPRETER_HIDDEN_SIZE, device=device)
        thinker_memory = torch.zeros(batch_size, BrainConfig.THINKER_MEMORY_SIZE, device=device)
        central_hidden = torch.zeros(1, batch_size, BrainConfig.WRITER_HIDDEN_SIZE, device=device)
        thalamus_hidden = torch.zeros(1, batch_size, BrainConfig.THALAMUS_HIDDEN_SIZE, device=device)
        mood_hidden = torch.zeros(1, batch_size, 256, device=device)
        
        return {
            'cmds': (reader_cmd, thinker_cmd, writer_cmd),
            'outs': (reader_out, thinker_out, writer_out),
            'sub_orders': (scout_orders, cursor_orders),
            'hiddens': (interpreter_hidden, thinker_memory, central_hidden, thalamus_hidden, mood_hidden),
            'mood': torch.zeros(batch_size, 256, device=device)
        }

    def forward(self, 
                reader_text_windows, 
                writer_text_windows,
                state):
        """
        One time step of the Brain.
        
        Args:
            reader_text_windows: (Batch, NumScouts, WinSize)
            writer_text_windows: (Batch, NumCursors, WinSize)
            state: Dictionary from init_state or previous step
            
        Returns:
            actions: Dict of logits (moves, writes, etc)
            next_state: Updated state dict
        """
        # Unpack State
        (cmd_reader, cmd_thinker, cmd_writer) = state['cmds']
        (out_reader, out_thinker, out_writer) = state['outs']
        (scout_orders, cursor_orders) = state['sub_orders']
        (h_interp, h_think, h_central, h_thal, h_mood) = state['hiddens']
        
        # Extract current token for Mood GRU (Last char seen by Cursor 0)
        # writer_text_windows: (Batch, NumCursors, WinSize)
        current_token_idx = writer_text_windows[:, 0, -1] # (Batch,)
        
        # 1. Thalamus delegates (CEO runs first to direct traffic based on previous step reports)
        # Alternatively, run components then Thalamus.
        # Let's run Thalamus first using "Previous Reports".
        new_cmd_reader, new_cmd_thinker, new_cmd_writer, shutdown_logits, new_h_thal, mood_vector, new_h_mood = \
            self.thalamus(out_reader, out_thinker, out_writer, current_token_idx, h_thal, h_mood)
            
        # 2. Reader Thing
        scout_moves, new_out_reader, new_scout_orders, scout_active, new_h_interp = \
            self.reader(reader_text_windows, new_cmd_reader, scout_orders, h_interp)
            
        # 3. Thinker Thing
        new_out_thinker, new_h_think = self.thinker(new_cmd_thinker, h_think)
        
        # 4. Writer Thing
        cursor_moves, cursor_acts, cursor_chars, cursor_offsets, new_out_writer, new_cursor_orders, send_logit, new_h_central = \
            self.writer(writer_text_windows, new_cmd_writer, cursor_orders, h_central)
            
        # Pack Next State
        next_state = {
            'cmds': (new_cmd_reader, new_cmd_thinker, new_cmd_writer),
            'outs': (new_out_reader, new_out_thinker, new_out_writer),
            'sub_orders': (new_scout_orders, new_cursor_orders),
            'hiddens': (new_h_interp, new_h_think, new_h_central, new_h_thal, new_h_mood),
            'mood': mood_vector
        }
        
        actions = {
            'scout_moves': scout_moves,       # (B, 5, 5)
            'scout_active': scout_active,     # (B, 5)
            'cursor_moves': cursor_moves,     # (B, 10, 5)
            'cursor_actions': cursor_acts,    # (B, 10, 4)
            'cursor_chars': cursor_chars,     # (B, 10, Vocab)
            'cursor_offsets': cursor_offsets, # (B, 10, WinSize+1)
            'send_logit': send_logit,         # (B, 1)
            'shutdown': shutdown_logits       # (B, 3)
        }
        
        return actions, next_state

    def detach_state(self, state):
        """
        Detaches all tensors in the state dictionary from the computation graph.
        Useful for TBPTT (Truncated Backpropagation Through Time).
        """
        new_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                new_state[k] = v.detach()
            elif isinstance(v, tuple):
                new_state[k] = tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in v)
            elif isinstance(v, list):
                new_state[k] = [x.detach() if isinstance(x, torch.Tensor) else x for x in v]
            elif isinstance(v, dict):
                new_state[k] = self.detach_state(v)
            else:
                new_state[k] = v
        return new_state


# Verification Code (if run directly)
if __name__ == "__main__":
    brain = GregBrain()
    print("GregBrain initialized.")
    print(f"Params: {sum(p.numel() for p in brain.parameters())}")
    
    # Dummy forward pass
    bs = 2
    device = torch.device('cpu')
    state = brain.init_state(bs, device)
    
    # Fake windows (random indices)
    r_wins = torch.randint(0, 256, (bs, BrainConfig.SCOUT_COUNT, BrainConfig.SCOUT_WINDOW_SIZE))
    w_wins = torch.randint(0, 256, (bs, BrainConfig.CURSOR_COUNT, BrainConfig.CURSOR_WINDOW_SIZE))
    
    actions, next_state = brain(r_wins, w_wins, state)
    print("Forward pass successful.")
    print("Action keys:", actions.keys())
