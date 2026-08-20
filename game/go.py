"""
Minimal Go implementation for 13x13 board.
"""


import numpy as np

from config import BOARD_SIZE, NUM_ACTIONS, PASS


class GoGame:
    """Minimal Go implementation for 13x13."""
    
    EMPTY, BLACK, WHITE = 0, 1, 2
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.to_play = self.BLACK
        self.ko = None
        self.passes = 0
        self.history = []
    
    def copy(self):
        g = GoGame()
        g.board = self.board.copy()
        g.to_play = self.to_play
        g.ko = self.ko
        g.passes = self.passes
        g.history = self.history.copy()
        return g
    
    def _opponent(self, color):
        return self.WHITE if color == self.BLACK else self.BLACK
    
    def _neighbors(self, x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                yield nx, ny
    
    def _get_group(self, x, y):
        """Returns (group_stones, liberties) for stone at (x,y)."""
        color = self.board[y, x]
        group, liberties = set(), set()
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group:
                continue
            group.add((cx, cy))
            for nx, ny in self._neighbors(cx, cy):
                if self.board[ny, nx] == self.EMPTY:
                    liberties.add((nx, ny))
                elif self.board[ny, nx] == color and (nx, ny) not in group:
                    stack.append((nx, ny))
        return group, liberties
    
    def is_legal(self, action: int) -> bool:
        if action == PASS:
            return True
        x, y = action % BOARD_SIZE, action // BOARD_SIZE
        if self.board[y, x] != self.EMPTY:
            return False
        if self.ko == (x, y):
            return False
        
        # Check suicide
        self.board[y, x] = self.to_play
        has_liberty = False
        captures = False
        
        _, my_libs = self._get_group(x, y)
        if my_libs:
            has_liberty = True
        else:
            for nx, ny in self._neighbors(x, y):
                if self.board[ny, nx] == self._opponent(self.to_play):
                    _, opp_libs = self._get_group(nx, ny)
                    if not opp_libs:
                        captures = True
                        break
        
        self.board[y, x] = self.EMPTY
        return has_liberty or captures
    
    def legal_moves(self) -> list[int]:
        return [a for a in range(NUM_ACTIONS) if self.is_legal(a)]
    
    def play(self, action: int) -> bool:
        if not self.is_legal(action):
            return False
        
        self.history.append(self.board.copy())
        
        if action == PASS:
            self.passes += 1
            self.ko = None
            self.to_play = self._opponent(self.to_play)
            return True
        
        self.passes = 0
        x, y = action % BOARD_SIZE, action // BOARD_SIZE
        self.board[y, x] = self.to_play
        
        # Capture opponent stones
        captured = []
        for nx, ny in self._neighbors(x, y):
            if self.board[ny, nx] == self._opponent(self.to_play):
                group, libs = self._get_group(nx, ny)
                if not libs:
                    captured.extend(group)
                    for gx, gy in group:
                        self.board[gy, gx] = self.EMPTY
        
        # Ko detection
        self.ko = None
        if len(captured) == 1:
            group, libs = self._get_group(x, y)
            if len(group) == 1 and len(libs) == 1:
                self.ko = captured[0]
        
        self.to_play = self._opponent(self.to_play)
        return True
    
    def is_terminal(self) -> bool:
        return self.passes >= 2
    
    def score(self) -> float:
        """Area scoring. Returns Black - White - Komi."""
        black, white = 0, 0
        visited = set()
        
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y, x] == self.BLACK:
                    black += 1
                elif self.board[y, x] == self.WHITE:
                    white += 1
                elif (x, y) not in visited:
                    # Flood fill empty region
                    region, borders = [], set()
                    stack = [(x, y)]
                    while stack:
                        cx, cy = stack.pop()
                        if (cx, cy) in visited:
                            continue
                        if self.board[cy, cx] != self.EMPTY:
                            borders.add(self.board[cy, cx])
                            continue
                        visited.add((cx, cy))
                        region.append((cx, cy))
                        for nx, ny in self._neighbors(cx, cy):
                            stack.append((nx, ny))
                    
                    if borders == {self.BLACK}:
                        black += len(region)
                    elif borders == {self.WHITE}:
                        white += len(region)
        
        return black - white - 6.5  # Komi
    
    def winner(self) -> int:
        """1=Black, -1=White, 0=Draw."""
        s = self.score()
        return 1 if s > 0 else (-1 if s < 0 else 0)
    
    def to_tensor(self) -> np.ndarray:
        """4 planes: own stones, opponent stones, empty, color to play."""
        own = self.to_play
        opp = self._opponent(own)
        planes = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[0] = (self.board == own)
        planes[1] = (self.board == opp)
        planes[2] = (self.board == self.EMPTY)
        planes[3] = 1.0 if own == self.BLACK else 0.0
        return planes
