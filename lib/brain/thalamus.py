import torch
import torch.nn as nn
from .config import BrainConfig

class Thalamus(nn.Module):
    """
    The CEO.
    Connects to Reader, Thinker, Writer.
    Delegates via vectors.
    """
    def __init__(self):
        super().__init__()
        
        # Inputs:
        # 1. From Reader (Interpreter)
        # 2. From Thinker
        # 3. From Writer (Central)
        input_size = (3 * BrainConfig.VECTOR_SIZE)
        
        self.rnn = nn.GRU(input_size, BrainConfig.THALAMUS_HIDDEN_SIZE, batch_first=True)
        
        # Outputs:
        # 1. To Reader
        # 2. To Thinker
        # 3. To Writer
        # 4. Shutdown signals (Reader, Thinker, Writer) - 3 logits
        self.to_reader = nn.Linear(BrainConfig.THALAMUS_HIDDEN_SIZE, BrainConfig.VECTOR_SIZE)
        self.to_thinker = nn.Linear(BrainConfig.THALAMUS_HIDDEN_SIZE, BrainConfig.VECTOR_SIZE)
        self.to_writer = nn.Linear(BrainConfig.THALAMUS_HIDDEN_SIZE, BrainConfig.VECTOR_SIZE)
        
        self.shutdown_logits = nn.Linear(BrainConfig.THALAMUS_HIDDEN_SIZE, 3)

    def forward(self, reader_vec, thinker_vec, writer_vec, hidden_state=None):
        """
        Args:
            reader_vec: (Batch, VectorSize)
            thinker_vec: (Batch, VectorSize)
            writer_vec: (Batch, VectorSize)
        """
        combined = torch.cat([reader_vec, thinker_vec, writer_vec], dim=1)
        
        output, next_hidden = self.rnn(combined.unsqueeze(1), hidden_state)
        output = output.squeeze(1)
        
        cmd_reader = self.to_reader(output)
        cmd_thinker = self.to_thinker(output)
        cmd_writer = self.to_writer(output)
        
        shutdown = self.shutdown_logits(output)
        
        return cmd_reader, cmd_thinker, cmd_writer, shutdown, next_hidden
