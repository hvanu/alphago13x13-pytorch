"""
SGF file parser for Go games.
"""

import re
from pathlib import Path

from config import BOARD_SIZE, PASS


def parse_sgf(content: str) -> list[int]:
    """Parse SGF content to list of moves."""
    moves = []
    for m in re.finditer(r';([BW])\[([a-z]{0,2})\]', content):
        coord = m.group(2)
        if not coord or coord == 'tt':
            moves.append(PASS)
        else:
            x, y = ord(coord[0]) - ord('a'), ord(coord[1]) - ord('a')
            if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                moves.append(y * BOARD_SIZE + x)
            else:
                return []  # Move outside board
    return moves


def load_sgf_files(paths: list[Path]) -> list[list[int]]:
    """Load SGF files and return list of games (each game is list of moves)."""
    games = []
    for path in paths:
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            # Accept 13x13 games or small unspecified games
            if 'SZ[13]' in content or ('SZ[' not in content and len(content) < 5000):
                moves = parse_sgf(content)
                if len(moves) >= 20:
                    games.append(moves)
        except Exception:
            pass
    return games
