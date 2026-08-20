"""
Self-play training and reinforcement learning.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_BATCH_SIZE, DEFAULT_MCTS_SIMS, DEFAULT_REPLAY_BUFFER_SIZE
from game import GoGame
from mcts import MCTS


class ReplayBuffer(Dataset):
    """Replay buffer for self-play samples."""
    
    def __init__(self, maxlen: int = DEFAULT_REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=maxlen)
    
    def add(self, samples: list[tuple]):
        self.buffer.extend(samples)
    
    def __len__(self): 
        return len(self.buffer)
    
    def __getitem__(self, i):
        s, p, v = self.buffer[i]
        return torch.from_numpy(s), torch.from_numpy(p), torch.tensor(v)


def self_play_game(model: torch.nn.Module, mcts_sims: int = DEFAULT_MCTS_SIMS, 
                   device: torch.device = None) -> list[tuple]:
    """
    Play one self-play game using MCTS.
    
    Args:
        model: Neural network model
        mcts_sims: Number of MCTS simulations per move
        device: Device to run on
    
    Returns:
        List of (state, mcts_policy, value) samples
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    mcts = MCTS(model, simulations=mcts_sims)
    game = GoGame()
    history = []
    
    move_num = 0
    while not game.is_terminal() and move_num < 200:
        temp = 1.0 if move_num < 20 else 0.1
        policy = mcts.search(game, temperature=temp)
        state_tensor = game.to_tensor()
        history.append((state_tensor, policy, game.to_play))
        
        if temp > 0.5:
            action = torch.multinomial(policy, 1).item()
        else:
            action = torch.argmax(policy).item()
        game.play(action)
        move_num += 1
    
    # Assign final values (convert to numpy only at the end)
    winner = game.winner()
    samples = []
    for state, policy, player in history:
        value = float(winner) if player == GoGame.BLACK else -float(winner)
        # Convert to numpy for storage in replay buffer
        state_np = state if isinstance(state, np.ndarray) else state
        policy_np = policy.cpu().numpy()
        samples.append((state_np, policy_np, np.float32(value)))
    
    return samples


def train_selfplay(module, iterations: int = 10, games_per_iter: int = 20,
                   epochs_per_iter: int = 3, mcts_sims: int = DEFAULT_MCTS_SIMS, 
                   batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Stage 2: Self-play training.
    
    Args:
        module: GoModule to train
        iterations: Number of training iterations
        games_per_iter: Number of self-play games per iteration
        epochs_per_iter: Number of training epochs per iteration
        mcts_sims: Number of MCTS simulations per move
        batch_size: Batch size for training
    
    Returns:
        Trained GoModule
    """
    print("\n" + "="*50)
    print("STAGE 2: Self-play training")
    print("="*50)
    
    buffer = ReplayBuffer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module.to(device)
    
    best_loss = float('inf')
    best_ckpt_path = None
    
    for it in range(iterations):
        print(f"\n--- Iteration {it+1}/{iterations} ---")
        
        # Self-play
        print(f"Generating {games_per_iter} self-play games...")
        for g in range(games_per_iter):
            samples = self_play_game(module.net, mcts_sims=mcts_sims, device=device)
            buffer.add(samples)
            if (g + 1) % 5 == 0:
                print(f"  Completed {g+1} games, buffer size: {len(buffer)}")
        
        # Train
        if len(buffer) >= batch_size:
            loader = DataLoader(buffer, batch_size=batch_size, shuffle=True, 
                              num_workers=4, persistent_workers=True)
            
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor='loss',
                mode='min',
                save_top_k=1,
                filename=f'selfplay-iter{it+1:02d}-{{epoch:02d}}-{{loss:.4f}}',
                verbose=False
            )
            
            trainer = pl.Trainer(
                max_epochs=epochs_per_iter,
                accelerator='auto',
                devices=1,
                enable_progress_bar=True,
                callbacks=[checkpoint_callback],
                logger=False,
            )
            trainer.fit(module, loader)
            
            # Track best checkpoint path across all iterations.
            if checkpoint_callback.best_model_score is not None:
                iter_best = float(checkpoint_callback.best_model_score)
                if iter_best < best_loss:
                    best_loss = iter_best
                    best_ckpt_path = checkpoint_callback.best_model_path
                    print(f"  New best loss: {best_loss:.4f}")
    
    # Load best model from all iterations.
    if best_ckpt_path:
        print(f"\nLoading best model from self-play (loss: {best_loss:.4f})")
        from .go_nn_lightning_module import GoModule
        best_module = GoModule.load_from_checkpoint(Path(best_ckpt_path))
        module.net.load_state_dict(best_module.net.state_dict())
    
    return module
