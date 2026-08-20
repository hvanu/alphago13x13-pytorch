"""Debug MCTS visit counts."""

import numpy as np
import torch

from main import BOARD_SIZE, MCTS, GoGame, GoNetwork, MCTSNode

# Load model
model = GoNetwork()
state_dict = torch.load('go_model.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Test with MCTS
game = GoGame()

# Manual MCTS to debug
mcts = MCTS(model, simulations=100)

root = MCTSNode()
with torch.no_grad():
    policy, _ = mcts.evaluate(game)
    
    print(f"Total legal moves: {len(game.legal_moves())}")
    print(f"MCTS simulations: {mcts.simulations}")
    
    for a in game.legal_moves():
        root.children[a] = MCTSNode(prior=policy[a].item())
    
    # Run simulations
    for sim in range(mcts.simulations):
        node, sim_game = root, game.copy()
        path = [node]
        
        # Select
        while node.children and not sim_game.is_terminal():
            action = max(node.children.keys(), 
                       key=lambda a: node.children[a].ucb(node.visits))
            node = node.children[action]
            path.append(node)
            sim_game.play(action)
        
        # Expand & evaluate
        if not sim_game.is_terminal():
            policy, value = mcts.evaluate(sim_game)
            for a in sim_game.legal_moves():
                node.children[a] = MCTSNode(prior=policy[a].item())
            value = value.item()
        else:
            value = sim_game.winner()
            if sim_game.to_play == GoGame.WHITE:
                value = -value
        
        # Backup
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value
    
    # Check visit distribution
    print(f"\nRoot visits: {root.visits}")
    visit_counts = [(a, child.visits, child.value()) for a, child in root.children.items()]
    visit_counts.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 moves by visit count:")
    for i, (action, visits, value) in enumerate(visit_counts[:10]):
        y, x = divmod(action, BOARD_SIZE)
        print(f"  {i+1}. ({x}, {y}) - visits: {visits}, value: {value:.3f}")
    
    print("\nBottom 10 moves by visit count:")
    for i, (action, visits, value) in enumerate(visit_counts[-10:]):
        y, x = divmod(action, BOARD_SIZE)
        print(f"  {i+1}. ({x}, {y}) - visits: {visits}, value: {value:.3f}")
    
    # Show distribution of visits
    all_visits = [child.visits for child in root.children.values()]
    print("\nVisit statistics:")
    print(f"  Min: {min(all_visits)}, Max: {max(all_visits)}, Mean: {np.mean(all_visits):.2f}")
    print(f"  Std: {np.std(all_visits):.2f}")
