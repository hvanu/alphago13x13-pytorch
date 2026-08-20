"""
Monte Carlo Tree Search implementation for Go.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from config import NUM_ACTIONS


class MCTSNode:
    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.children: dict = {}
        self.visits = 0
        self.value_sum = 0.0
    
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0
    
    def ucb(self, parent_visits: int, c: float = 1.5) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value() + c * self.prior * math.sqrt(parent_visits) / (1 + self.visits)


class MCTS:
    def __init__(self, model: nn.Module, simulations: int = 800):
        self.model = model
        self.simulations = simulations
        self.device = next(model.parameters()).device
    
    @torch.no_grad()
    def evaluate(self, game):
        """Evaluate a game position using the neural network."""
        x = torch.from_numpy(game.to_tensor()).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(x)
        policy = F.softmax(policy_logits, dim=1).squeeze()
        return policy, value.squeeze()
    
    def search(self, game, temperature: float = 1.0) -> torch.Tensor:
        """
        Run MCTS search from the given game position.
        
        Args:
            game: GoGame instance
            temperature: Temperature for action selection (0 = greedy, higher = more exploration)
        
        Returns:
            Probability distribution over actions based on MCTS visit counts
        """
        root = MCTSNode()
        policy, _ = self.evaluate(game)
        
        # Convert policy to list once for faster indexing (avoid repeated .item() calls)
        policy_list = policy.tolist()
        legal_moves = game.legal_moves()
        for a in legal_moves:
            root.children[a] = MCTSNode(prior=policy_list[a])
        
        for _ in range(self.simulations):
            node, sim_game = root, game.copy()
            path = [node]
            
            # Select
            while node.children and not sim_game.is_terminal():
                # Cache parent_visits to avoid repeated attribute access
                parent_visits = node.visits
                # Find best action using UCB
                best_action = None
                best_ucb = -float('inf')
                for action, child in node.children.items():
                    ucb_val = child.ucb(parent_visits)
                    if ucb_val > best_ucb:
                        best_ucb = ucb_val
                        best_action = action
                
                node = node.children[best_action]
                path.append(node)
                sim_game.play(best_action)
            
            # Expand & evaluate
            if not sim_game.is_terminal():
                policy, value = self.evaluate(sim_game)
                # Convert to list once for faster indexing
                policy_list = policy.tolist()
                legal_moves = sim_game.legal_moves()
                for a in legal_moves:
                    node.children[a] = MCTSNode(prior=policy_list[a])
                value = float(value.item())
            else:
                value = float(sim_game.winner())
                # Import here to avoid circular dependency
                from game import GoGame
                if sim_game.to_play == GoGame.WHITE:
                    value = -value
            
            # Backup
            for node in reversed(path):
                node.visits += 1
                node.value_sum += value
                value = -value
        
        # Extract policy from visit counts
        visits = torch.zeros(NUM_ACTIONS, dtype=torch.float32, device=self.device)
        for a, child in root.children.items():
            visits[a] = child.visits
        
        if temperature == 0:
            probs = torch.zeros_like(visits)
            probs[torch.argmax(visits)] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / (visits.sum() + 1e-8)
        
        return probs
