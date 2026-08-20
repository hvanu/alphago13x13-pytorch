"""FastAPI service for Go game state and move generation."""

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BOARD_SIZE, NUM_ACTIONS
from game import GoGame
from mcts import MCTS
from model import GoNetwork


# Models for API requests/responses
class Position(BaseModel):
    x: int
    y: int

class MoveRequest(BaseModel):
    game_id: str
    x: int = 0
    y: int = 0
    pass_move: bool = False

class GameState(BaseModel):
    id: str
    board: list[list[int]]
    to_play: int
    passes: int
    ko: Position | None = None

class MoveResponse(BaseModel):
    game: GameState
    ai_move: dict[str, Any] | None = None
    game_over: bool = False


class GoGameManager:
    """Manages Go games and AI inference."""
    
    def __init__(self, model_path: str, simulations: int = 100):
        self.games: dict[str, GoGame] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info("initializing model on %s", self.device)

        self.model = GoNetwork()
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info("model loaded from %s", model_path)
        self.mcts = MCTS(self.model, simulations=simulations)
        logger.info("MCTS ready (%d simulations)", simulations)
    
    def create_game(self) -> str:
        """Create a new game and return its ID."""
        game_id = f"game-{uuid.uuid4()}"
        self.games[game_id] = GoGame()
        logger.info(f"Created new game: {game_id}")
        return game_id
    
    def get_game(self, game_id: str) -> GoGame | None:
        """Get a game by ID."""
        return self.games.get(game_id)
    
    def game_to_dict(self, game: GoGame) -> dict[str, Any]:
        """Convert game state to dictionary."""
        ko_pos = None
        if hasattr(game, 'ko') and game.ko is not None:
            ko_y, ko_x = divmod(game.ko, BOARD_SIZE)
            ko_pos = {"x": int(ko_x), "y": int(ko_y)}
        
        return {
            "board": game.board.tolist(),
            "to_play": int(game.to_play),
            "passes": game.passes,
            "ko": ko_pos
        }
    
    def is_legal(self, game: GoGame, action: int) -> bool:
        """Check if a move is legal."""
        if action == NUM_ACTIONS - 1:  # Pass move
            return True
        return game.is_legal(action)
    
    def play_move(self, game: GoGame, x: int, y: int, is_pass: bool) -> bool:
        """
        Play a move on the board.
        Returns True if successful, False otherwise.
        """
        if is_pass:
            action = NUM_ACTIONS - 1
        else:
            if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                logger.warning("out-of-bounds move attempted: (%d, %d)", x, y)
                return False
            action = y * BOARD_SIZE + x
        
        if not self.is_legal(game, action):
            logger.warning("illegal move attempted: action=%d", action)
            return False
        
        game.play(action)
        return True
    
    def get_ai_move(self, game: GoGame, temperature: float = 0.1) -> tuple[int, int, bool]:
        """
        Get AI's move using MCTS.
        Returns (x, y, is_pass).
        """

        policy = self.mcts.search(game, temperature=temperature)

        action = torch.argmax(policy).item()
        
        if action == NUM_ACTIONS - 1:  # Pass
            logger.info("AI move: pass")
            return 0, 0, True
        
        x = action % BOARD_SIZE
        y = action // BOARD_SIZE
        logger.info("AI move: (%d, %d)", x, y)
        return x, y, False


# Global game manager (initialized on startup)
game_manager: GoGameManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global game_manager
    
    logger.info("starting Go AI server")
    
    # Get model path from environment or use default
    model_path = os.getenv("MODEL_PATH", "./go_model.pt")
    logger.info("model path: %s", model_path)
    
    # If it's a Lightning checkpoint, we need the state dict
    if model_path.endswith('.ckpt'):
        # Prefer exported inference weights when available.
        pt_path = "../go_model_pretrain.pt"
        logger.info("checkpoint provided, probing for %s", pt_path)
        if os.path.exists(pt_path):
            model_path = pt_path
            logger.info("using exported weights: %s", pt_path)
        else:
            logger.warning("checkpoint provided but no .pt export found; export may be required")
    
    if not os.path.exists(model_path):
        logger.error("model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    simulations = int(os.getenv("MCTS_SIMULATIONS", "800"))
    game_manager = GoGameManager(model_path, simulations=simulations)
    logger.info("server initialization complete")
    
    yield
    
    # Cleanup on shutdown (if needed)


app = FastAPI(title="Go AI Server", lifespan=lifespan)


@app.post("/api/game/new")
async def new_game() -> dict[str, Any]:
    if game_manager is None:
        raise HTTPException(status_code=500, detail="Game manager not initialized")
    
    game_id = game_manager.create_game()
    game = game_manager.get_game(game_id)
    
    return {
        "id": game_id,
        **game_manager.game_to_dict(game)
    }


@app.get("/api/game/{game_id}")
async def get_game(game_id: str) -> dict[str, Any]:
    if game_manager is None:
        raise HTTPException(status_code=500, detail="Game manager not initialized")
    
    game = game_manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return {
        "id": game_id,
        **game_manager.game_to_dict(game)
    }


@app.post("/api/game/move")
async def make_move(request: MoveRequest) -> MoveResponse:
    """Make a move (player then AI)."""
    logger.info(
        "move request: game=%s move=(%d,%d) pass=%s",
        request.game_id,
        request.x,
        request.y,
        request.pass_move,
    )
    
    if game_manager is None:
        raise HTTPException(status_code=500, detail="Game manager not initialized")
    
    game = game_manager.get_game(request.game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")
    

    success = game_manager.play_move(game, request.x, request.y, request.pass_move)
    if not success:
        logger.error("rejecting illegal move: (%d, %d)", request.x, request.y)
        raise HTTPException(status_code=400, detail="Illegal move")
    
    if game.passes >= 2:
        logger.info("game over after player move (double pass)")
        return MoveResponse(
            game=GameState(
                id=request.game_id,
                **game_manager.game_to_dict(game)
            ),
            game_over=True
        )
    
    ai_x, ai_y, ai_pass = game_manager.get_ai_move(game)
    
    game_manager.play_move(game, ai_x, ai_y, ai_pass)
    logger.info("turn complete; passes=%d", game.passes)
    
    return MoveResponse(
        game=GameState(
            id=request.game_id,
            **game_manager.game_to_dict(game)
        ),
        ai_move={"x": ai_x, "y": ai_y, "pass": ai_pass},
        game_over=game.passes >= 2
    )


frontend_dir = Path(__file__).parent.parent / "frontend/dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
