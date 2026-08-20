"""
Neural network model for Go.
"""

from .network import GoNetwork, PolicyHead, ResBlock, ValueHead

__all__ = ['GoNetwork', 'PolicyHead', 'ResBlock', 'ValueHead']
