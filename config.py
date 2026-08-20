"""
Configuration constants for Go AlphaZero training.
"""

# Board configuration
BOARD_SIZE = 13
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1  # +1 for pass
PASS = BOARD_SIZE * BOARD_SIZE

# Training hyperparameters
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 64
DEFAULT_MCTS_SIMS = 50
DEFAULT_REPLAY_BUFFER_SIZE = 30000

# Network architecture
DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 6
INPUT_CHANNELS = 4  # own stones, opponent stones, empty, color to play
