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

        # Global Mood GRU
        self.embed = nn.Embedding(BrainConfig.VOCAB_SIZE, BrainConfig.EMBED_DIM)
        self.mood_gru = nn.GRU(BrainConfig.EMBED_DIM, 256, batch_first=True)
        self.mood_to_context = nn.Linear(256, BrainConfig.VECTOR_SIZE) # Project to context vector if needed

    def forward(self, reader_vec, thinker_vec, writer_vec, current_token_idx, hidden_state=None, mood_hidden=None):
        """
        Args:
            reader_vec: (Batch, VectorSize)
            thinker_vec: (Batch, VectorSize)
            writer_vec: (Batch, VectorSize)
            current_token_idx: (Batch,) LongTensor - Current token being processed
        """
        combined = torch.cat([reader_vec, thinker_vec, writer_vec], dim=1)
        
        output, next_hidden = self.rnn(combined.unsqueeze(1), hidden_state)
        output = output.squeeze(1)
        
        # Mood GRU Step
        # Embed token
        if current_token_idx is not None:
            # Clamp to vocab size just in case
            safe_idx = torch.clamp(current_token_idx, 0, BrainConfig.VOCAB_SIZE - 1)
            embeds = self.embed(safe_idx).unsqueeze(1) # (Batch, 1, Embed)
            mood_out, next_mood_hidden = self.mood_gru(embeds, mood_hidden)
            mood_vector = mood_out.squeeze(1) # (Batch, 256)
        else:
            # Zero mood if no token (e.g. init)
            device = reader_vec.device
            mood_vector = torch.zeros(reader_vec.size(0), 256, device=device)
            next_mood_hidden = mood_hidden

        cmd_reader = self.to_reader(output)
        cmd_thinker = self.to_thinker(output)
        cmd_writer = self.to_writer(output)
        
        shutdown = self.shutdown_logits(output)
        
        return cmd_reader, cmd_thinker, cmd_writer, shutdown, next_hidden, mood_vector, next_mood_hidden

