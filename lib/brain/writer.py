import torch
import torch.nn as nn
from .config import BrainConfig
from .modules import WindowScoutBase, MLP

class WriterCursor(WindowScoutBase):
    """
    Writer Cursor:
    - Similar to Scout (sees text).
    - Can Write, Change, or Delete characters.
    """
    def __init__(self):
        super().__init__(hidden_size=BrainConfig.CURSOR_HIDDEN_SIZE, window_size=BrainConfig.CURSOR_WINDOW_SIZE)
        
        # Extra capability: Writing
        # Actions:
        # 1. Write Character (Vocabulary Size)
        # 2. Delete Character (Binary/Logit)
        # 3. No Op (implicit in others or separate?)
        
        # Let's combine this into a single "Action" head or multiple heads.
        # User said: "write a character, change a character or delete a character"
        # "Change" is effectively "Write" at current position.
        # "Write" usually means insert.
        # "Delete" means remove.
        
        # We need to output:
        # 1. Action Type Logits: [NoOp, Insert, Overwrite, Delete] (4 classes)
        # 2. Character Logits: [Vocab Size] (which char to insert/overwrite)
        
        # We reuse the MLP hidden state from WindowScoutBase
        self.action_type_logits = nn.Linear(BrainConfig.CURSOR_HIDDEN_SIZE, 4)
        self.char_logits = nn.Linear(BrainConfig.CURSOR_HIDDEN_SIZE, BrainConfig.VOCAB_SIZE)
        # Offset: 0 to WindowSize (inclusive) -> WindowSize + 1 classes
        self.offset_logits = nn.Linear(BrainConfig.CURSOR_HIDDEN_SIZE, BrainConfig.CURSOR_WINDOW_SIZE + 1)

    def forward(self, text_window_indices, command_vector):
        """
        Returns:
            out_vector, move_logits, action_type_logits, char_logits, offset_logits
        """
        # Get base scout outputs
        # Note: We need the hidden state, but WindowScoutBase.forward returns final outputs.
        # I'll override forward or modify base. 
        # For cleanliness, I'll copy the logic since it's short, or better, split base.
        # Let's just copy logic to avoid modifying base and breaking Reader.
        
        embeds = self._embed(text_window_indices)
        embeds_flat = embeds.view(embeds.size(0), -1)
        combined = torch.cat([embeds_flat, command_vector], dim=1)
        hidden = self.mlp(combined)
        
        out_vector = self.to_vector(hidden)
        move_logits = self.move_logits(hidden)
        
        # Writer specific
        action_type = self.action_type_logits(hidden)
        char_val = self.char_logits(hidden)
        offset_val = self.offset_logits(hidden)
        
        return out_vector, move_logits, action_type, char_val, offset_val

class CentralWriter(nn.Module):
    """
    Central Writer:
    - Interprets orders from Thalamus.
    - Controls Cursors.
    - Decides if response is OK to send.
    """
    def __init__(self):
        super().__init__()
        self.num_cursors = BrainConfig.CURSOR_COUNT
        self.vector_size = BrainConfig.VECTOR_SIZE
        
        # Input: Vectors from Cursors + Thalamus Command
        input_size = (self.num_cursors * self.vector_size) + self.vector_size
        
        self.rnn = nn.GRU(input_size, BrainConfig.WRITER_HIDDEN_SIZE, batch_first=True)
        
        # Heads
        self.to_cursors = nn.Linear(BrainConfig.WRITER_HIDDEN_SIZE, self.num_cursors * self.vector_size)
        self.to_thalamus = nn.Linear(BrainConfig.WRITER_HIDDEN_SIZE, self.vector_size)
        
        # Send Decision (Sigmoid logic, so 1 output logit)
        self.send_logit = nn.Linear(BrainConfig.WRITER_HIDDEN_SIZE, 1)

    def forward(self, cursor_vectors, thalamus_command, hidden_state=None):
        batch_size = cursor_vectors.size(0)
        cursors_flat = cursor_vectors.view(batch_size, -1)
        combined = torch.cat([cursors_flat, thalamus_command], dim=1)
        
        output, next_hidden = self.rnn(combined.unsqueeze(1), hidden_state)
        output = output.squeeze(1)
        
        cursor_orders_flat = self.to_cursors(output)
        cursor_orders = cursor_orders_flat.view(batch_size, self.num_cursors, self.vector_size)
        
        to_thalamus = self.to_thalamus(output)
        send_logit = self.send_logit(output)
        
        return cursor_orders, to_thalamus, send_logit, next_hidden

class WriterThing(nn.Module):
    def __init__(self):
        super().__init__()
        self.cursors = nn.ModuleList([WriterCursor() for _ in range(BrainConfig.CURSOR_COUNT)])
        self.central = CentralWriter()
        
    def forward(self, text_windows, thalamus_command, previous_cursor_orders, central_hidden):
        """
        Args:
            text_windows: (Batch, NumCursors, WindowSize)
        """
        batch_size = text_windows.size(0)
        cursor_vectors_list = []
        cursor_moves_list = []
        cursor_actions_list = []
        cursor_chars_list = []
        cursor_offsets_list = []
        
        for i, cursor in enumerate(self.cursors):
            win = text_windows[:, i, :]
            order = previous_cursor_orders[:, i, :]
            
            vec, moves, acts, chars, offsets = cursor(win, order)
            
            cursor_vectors_list.append(vec)
            cursor_moves_list.append(moves)
            cursor_actions_list.append(acts)
            cursor_chars_list.append(chars)
            cursor_offsets_list.append(offsets)
            
        cursor_vectors = torch.stack(cursor_vectors_list, dim=1)
        cursor_moves = torch.stack(cursor_moves_list, dim=1)
        cursor_actions = torch.stack(cursor_actions_list, dim=1)
        cursor_chars = torch.stack(cursor_chars_list, dim=1)
        cursor_offsets = torch.stack(cursor_offsets_list, dim=1)
        
        # Central logic
        new_orders, to_thal, send_logit, new_hidden = self.central(
            cursor_vectors, thalamus_command, central_hidden
        )
        
        return cursor_moves, cursor_actions, cursor_chars, cursor_offsets, to_thal, new_orders, send_logit, new_hidden
