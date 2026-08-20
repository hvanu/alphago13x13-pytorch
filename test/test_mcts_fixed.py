"""Test MCTS with more simulations."""

import torch
import torch.nn.functional as F

from main import BOARD_SIZE, MCTS, GoGame, GoNetwork

# Load model
model = GoNetwork()
state_dict = torch.load('go_model.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Test with more MCTS simulations
game = GoGame()
mcts = MCTS(model, simulations=800)

# First check what the network outputs
with torch.no_grad():
    state = torch.from_numpy(game.to_tensor()).unsqueeze(0)
    policy_logits, value = model(state)
    policy_probs = F.softmax(policy_logits, dim=1).squeeze()
    
    print('=== Raw Network Output ===')
    print(f'Policy probs: min={policy_probs.min():.6f}, max={policy_probs.max():.6f}')
    top5_vals, top5_idx = torch.topk(policy_probs, 5)
    print('Top 5 by network:')
    for i, (v, idx) in enumerate(zip(top5_vals, top5_idx)):
        y, x = divmod(idx.item(), BOARD_SIZE)
        print(f'  {i+1}. ({x}, {y}) - prob: {v.item():.4f}')
    
    print()
    print('=== After MCTS (800 simulations) ===')
    # Run MCTS
    policy = mcts.search(game, temperature=0.1)
    print(f'MCTS policy: min={policy.min():.6f}, max={policy.max():.6f}')
    print(f'MCTS policy sum: {policy.sum():.4f}')
    
    top5_vals, top5_idx = torch.topk(policy, 5)
    print('Top 5 by MCTS:')
    for i, (v, idx) in enumerate(zip(top5_vals, top5_idx)):
        y, x = divmod(idx.item(), BOARD_SIZE)
        print(f'  {i+1}. ({x}, {y}) - prob: {v.item():.4f}')
        
    print()
    print('Checking if MCTS found the same top move as network...')
    net_top = torch.argmax(policy_probs).item()
    mcts_top = torch.argmax(policy).item()
    net_y, net_x = divmod(net_top, BOARD_SIZE)
    mcts_y, mcts_x = divmod(mcts_top, BOARD_SIZE)
    print(f'Network top: ({net_x}, {net_y})')
    print(f'MCTS top: ({mcts_x}, {mcts_y})')
    
    if net_top == mcts_top:
        print('✓ MCTS agrees with network!')
    else:
        print('✗ MCTS disagrees with network (but this is okay - MCTS should improve on network)')
