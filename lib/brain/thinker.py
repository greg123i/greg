import torch
import torch.nn as nn
from .config import BrainConfig

class ThinkerThing(nn.Module):
    """
    Thinker Thing:
    - RNN with large recurrent output (memory).
    - Communicates with Thalamus.
    """
    def __init__(self):
        super().__init__()
        
        # Input: From Thalamus + Recurrent Memory
        self.input_size = BrainConfig.VECTOR_SIZE + BrainConfig.THINKER_MEMORY_SIZE
        
        # We can use a GRU or simple RNN. 
        # Since the user specified "2048 of its outputs as inputs", it implies the hidden state IS the memory.
        # Or it's an explicit feedback loop.
        # Let's use a GRU where hidden_size = MEMORY_SIZE (2048).
        
        self.hidden_size = BrainConfig.THINKER_MEMORY_SIZE
        self.rnn = nn.GRUCell(self.input_size, self.hidden_size)
        
        # Output to Thalamus
        self.to_thalamus = nn.Linear(self.hidden_size, BrainConfig.VECTOR_SIZE)

    def forward(self, thalamus_vector, prev_memory):
        """
        Args:
            thalamus_vector: (Batch, VectorSize)
            prev_memory: (Batch, MemorySize) - Acts as input and hidden state
            
        Returns:
            to_thalamus: (Batch, VectorSize)
            new_memory: (Batch, MemorySize)
        """
        # Combine input
        combined = torch.cat([thalamus_vector, prev_memory], dim=1)
        
        # RNN Step
        # Note: GRUCell expects (Batch, Input), Hidden
        # We use prev_memory as the hidden state
        new_memory = self.rnn(combined, prev_memory)
        
        # Generate output
        to_thal = self.to_thalamus(new_memory)
        
        return to_thal, new_memory
