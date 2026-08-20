"""
Export a Lightning checkpoint to plain PyTorch weights.
"""

import argparse
from pathlib import Path

import torch

from model import GoNetwork
from train import GoModule


def export_checkpoint(checkpoint_path: str, output_path: str):
    """
    Export only the network state dict from a Lightning checkpoint.
    """
    print(f"Loading Lightning checkpoint from: {checkpoint_path}")

    module = GoModule.load_from_checkpoint(checkpoint_path)
    model_state_dict = module.net.state_dict()

    torch.save(model_state_dict, output_path)

    print(f"✓ Model exported to: {output_path}")
    print(f"  Model size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

    print("\nVerifying export...")
    test_model = GoNetwork()
    test_model.load_state_dict(torch.load(output_path, map_location='cpu'))
    print("✓ Export verified - model loads successfully")


def main():
    parser = argparse.ArgumentParser(description="Export Lightning checkpoint to .pt file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="lightning_logs/version_0/checkpoints/epoch=9-step=1880.ckpt",
        help="Path to Lightning checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="go_model.pt",
        help="Output path for .pt file"
    )
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        for ckpt in Path("lightning_logs").rglob("*.ckpt"):
            print(f"  - {ckpt}")
        return 1

    export_checkpoint(args.checkpoint, args.output)

    print("\nYou can now use the exported model:")
    print(f"  MODEL_PATH={args.output} python backend/server.py")

    return 0


if __name__ == "__main__":
    exit(main())
