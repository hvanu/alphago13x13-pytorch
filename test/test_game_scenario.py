"""Test MCTS with a game in progress."""

import torch

from main import BOARD_SIZE, MCTS, GoGame, GoNetwork

# Load model
model = GoNetwork()
state_dict = torch.load('go_model.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Create a game with some moves
game = GoGame()
game.play(9 * BOARD_SIZE + 3)  # (3, 9) - the move from the logs

print('After playing at (3, 9):')
print(f'Current player: {game.to_play}')
print(f'Legal moves: {len(game.legal_moves())}')

# Test MCTS
mcts = MCTS(model, simulations=800)

with torch.no_grad():
    print('\n=== MCTS Result ===')
    policy = mcts.search(game, temperature=0.1)
    print(f'MCTS policy: min={policy.min():.6f}, max={policy.max():.6f}')
    
    top5_vals, top5_idx = torch.topk(policy, 5)
    print('Top 5 moves by MCTS:')
    for i, (v, idx) in enumerate(zip(top5_vals, top5_idx)):
        if idx.item() == 169:  # Pass move
            print(f'  {i+1}. PASS - prob: {v.item():.4f}')
        else:
            y, x = divmod(idx.item(), BOARD_SIZE)
            print(f'  {i+1}. ({x}, {y}) - prob: {v.item():.4f}')
    
    # Check if it's choosing (0,0)
    action_0_0 = policy[0].item()
    print(f'\nProbability of (0, 0): {action_0_0:.6f}')
    
    if action_0_0 > 0.05:
        print('⚠️  Still preferring (0, 0) - problem persists!')
    else:
        print('✓ (0, 0) has low probability - problem fixed!')
