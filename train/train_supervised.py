"""
Supervised pretraining from SGF files.
"""

import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from config import DEFAULT_BATCH_SIZE, NUM_ACTIONS
from game import GoGame, load_sgf_files


class SGFDataset(Dataset):
    """Dataset from SGF games."""
    
    def __init__(self, games: list[list[int]]):
        self.samples = []
        for moves in games:
            game = GoGame()
            for i, move in enumerate(moves):
                if game.is_legal(move):
                    state = game.to_tensor()
                    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                    policy[move] = 1.0
                    # Heuristic: alternating winner from last move perspective
                    value = 1.0 if (len(moves) - i) % 2 == 1 else -1.0
                    self.samples.append((state, policy, value))
                    game.play(move)
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, i):
        s, p, v = self.samples[i]
        return torch.from_numpy(s), torch.from_numpy(p), torch.tensor(v)


def pretrain(sgf_dir: Path | None, epochs: int = 1000, 
             batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Stage 1: Pretrain from SGF files.
    
    Args:
        sgf_dir: Directory containing SGF files
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained GoModule
    """
    from .go_nn_lightning_module import GoModule
    
    print("\n" + "="*50)
    print("STAGE 1: Pretraining from SGF files")
    print("="*50)
    
    games = []
    if sgf_dir:
        if not sgf_dir.exists():
            print(f"SGF directory {sgf_dir} does not exist.")
        else:
            sgf_files = list(sgf_dir.rglob('*.sgf'))
            games = load_sgf_files(sgf_files)
            print(f"Loaded {len(games)} games from {len(sgf_files)} files")
    
    if not games:
        print("No SGF games found. Generating random games for demo...")
        for _ in range(200):
            game = GoGame()
            moves = []
            for _ in range(60):
                legal = game.legal_moves()
                move = random.choice(legal)
                moves.append(move)
                game.play(move)
                if game.is_terminal():
                    break
            games.append(moves)
    
    dataset = SGFDataset(games)
    print(f"Training on {len(dataset)} positions")
    
    module = GoModule()
    
    # Save best model based on lowest loss
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='loss',
        mode='min',
        save_top_k=1,
        filename='pretrain-best-{epoch:02d}-{loss:.4f}',
        verbose=True
    )
    
    early_stop = pl.callbacks.EarlyStopping(
        monitor='loss',
        patience=20,
        mode='min',
        verbose=True
    )
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop],
    )
    trainer.fit(module, DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                   num_workers=4, persistent_workers=True))
    
    # Load best checkpoint
    if checkpoint_callback.best_model_path:
        print(f"\nLoading best pretrain checkpoint: {checkpoint_callback.best_model_path}")
        module = GoModule.load_from_checkpoint(checkpoint_callback.best_model_path)
    
    return module
