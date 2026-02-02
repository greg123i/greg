import torch
import torch.nn as nn
from .config import BrainConfig

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron helper.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):
    """
    Standard Multi-Head Attention Block with Residual Connection and LayerNorm.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (Batch, SeqLen, EmbedDim)
        Returns:
            x: (Batch, SeqLen, EmbedDim)
        """
        attn_out, _ = self.mha(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x

class WindowScoutBase(nn.Module):
    """
    Base class for Scouts and Cursors.
    Reads a window of text and a command vector, outputs a vector and movement logits.
    """
    def __init__(self, hidden_size, window_size=BrainConfig.SCOUT_WINDOW_SIZE, input_vector_size=BrainConfig.VECTOR_SIZE, output_vector_size=BrainConfig.VECTOR_SIZE):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = BrainConfig.EMBED_DIM
        
        # Text embedding (Nibble based: 2x 4-bit embeddings concatenated)
        self.nibble_embed_dim = self.embed_dim // 2
        self.nibble_embedding = nn.Embedding(BrainConfig.NIBBLE_VOCAB_SIZE, self.nibble_embed_dim)
        
        # Input processing: (Window * Embed) + Command Vector
        self.input_flat_size = (self.window_size * self.embed_dim) + input_vector_size
        
        # Core Logic
        self.mlp = MLP(self.input_flat_size, hidden_size, hidden_size)
        
        # Outputs
        self.to_vector = nn.Linear(hidden_size, output_vector_size)
        
        # Movement: -5, -1, 0 (stay), +1, +5
        # 5 classes
        self.move_logits = nn.Linear(hidden_size, 5) 

    def _embed(self, indices):
        """
        Embeds indices by splitting into high and low nibbles.
        Handles unknown tokens (>255) by masking to 8 bits.
        """
        # Ensure 0-255 range (Extended ASCII)
        indices = indices & 0xFF
        
        high_nib = (indices >> 4) & 0xF
        low_nib = indices & 0xF
        
        e_high = self.nibble_embedding(high_nib)
        e_low = self.nibble_embedding(low_nib)
        
        # Concatenate: (Batch, Window, EmbedDim)
        return torch.cat([e_high, e_low], dim=-1)

    def forward(self, text_window_indices, command_vector):
        """
        Args:
            text_window_indices: (Batch, WindowSize) - LongTensor of char indices
            command_vector: (Batch, VectorSize) - FloatTensor
        """
        # Embed text: (Batch, Window, Embed)
        embeds = self._embed(text_window_indices)
        
        # Flatten: (Batch, Window * Embed)
        embeds_flat = embeds.view(embeds.size(0), -1)
        
        # Concatenate with command
        combined = torch.cat([embeds_flat, command_vector], dim=1)
        
        # Process
        hidden = self.mlp(combined)
        
        # Outputs
        out_vector = self.to_vector(hidden)
        moves = self.move_logits(hidden)
        
        return out_vector, moves

