# Configuration for the Brain Architecture

class BrainConfig:
    # General
    VOCAB_SIZE = 256  # ASCII 0-255
    NIBBLE_VOCAB_SIZE = 16 # 4-bit nibbles (0-15)
    EMBED_DIM = 64    # Character embedding size
    
    # Vectors (Communication Channels)
    VECTOR_SIZE = 128 # Standard size for inter-module communication
    
    # Scout Specifics
    SCOUT_IDEA_SIZE = 20   # Output from Scout (Encoder)
    SCOUT_ORDER_SIZE = 12  # Input to Scout (from Interpreter)
    
    # Reader / Scouts
    SCOUT_COUNT = 3
    SCOUT_WINDOW_SIZE = 49
    SCOUT_HIDDEN_SIZE = 256
    INTERPRETER_HIDDEN_SIZE = 256
    
    # Thinker
    THINKER_COUNT = 1
    THINKER_MEMORY_SIZE = 1024  # Reduced for speed
    THINKER_HIDDEN_SIZE = 512
    
    # Writer / Cursors
    CURSOR_COUNT = 5
    CURSOR_WINDOW_SIZE = 25
    CURSOR_HIDDEN_SIZE = 256
    CURSOR_TOP_K = 5 # Selection size
    WRITER_HIDDEN_SIZE = 256
    
    # Thalamus (CEO)
    THALAMUS_HIDDEN_SIZE = 512
    
    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
