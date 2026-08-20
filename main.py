"""
Training entrypoint: SGF pretrain followed by self-play fine-tuning.
"""

from pathlib import Path

import torch

from train import pretrain, train_selfplay


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Go AI (13x13)')
    parser.add_argument('--sgf-dir', type=Path, default=None, help='Directory with SGF files')
    parser.add_argument('--pretrain-epochs', type=int, default=1000)
    parser.add_argument('--selfplay-iters', type=int, default=5)
    parser.add_argument('--games-per-iter', type=int, default=20)
    parser.add_argument('--mcts-sims', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('go_model.pt'))
    args = parser.parse_args()
    
    # Stage 1: supervised pretrain.
    module = pretrain(args.sgf_dir, epochs=args.pretrain_epochs, batch_size=args.batch_size)
    
    pretrain_output = args.output.parent / (args.output.stem + '_pretrain.pt')
    torch.save(module.net.state_dict(), pretrain_output)
    print(f"\n✓ Best pretrained model saved to {pretrain_output}")
    
    # Stage 2: self-play reinforcement learning.
    module = train_selfplay(
        module,
        iterations=args.selfplay_iters,
        games_per_iter=args.games_per_iter,
        mcts_sims=args.mcts_sims,
        batch_size=args.batch_size,
    )
    
    torch.save(module.net.state_dict(), args.output)
    print(f"\n✓ Best self-play model saved to {args.output}")


if __name__ == '__main__':
    main()


