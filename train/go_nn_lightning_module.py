"""
PyTorch Lightning module for Go model training.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from config import DEFAULT_LR
from model import GoNetwork


class GoModule(pl.LightningModule):
    def __init__(self, lr: float = DEFAULT_LR):
        super().__init__()
        self.save_hyperparameters()
        self.net = GoNetwork()
        self.ema_decay = 0.999
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        states, target_policy, target_value = batch
        policy_logits, value = self.net(states)
        
        policy_loss = F.cross_entropy(policy_logits, target_policy)
        value_loss = F.mse_loss(value.squeeze(-1), target_value)
        
        loss = 0.75 * policy_loss + 1.0 * value_loss
        
        self.log_dict({
            'loss': loss, 
            'p_loss': policy_loss, 
            'v_loss': value_loss,
            'lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }, prog_bar=True)
        
        return loss
     
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=self.hparams.lr * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
