# Go AlphaZero (13x13)

Python implementation of a small AlphaGo/AlphaZero-style Go system:

- game engine for 13x13 Go
- residual policy/value network (PyTorch)
- MCTS move search
- training pipeline (SGF pretrain + self-play)
- FastAPI backend serving game engine and TypeScript frontend with a very simple board
- training scripts

## Repository layout

```
.
├── main.py                     # Training entrypoint
├── config.py                   # Board and training constants
├── game/                       # Go rules + SGF parsing
├── model/                      # Neural network
├── mcts/                       # Tree search
├── train/                      # Lightning module + train loops
├── backend/server.py           # FastAPI game server
├── frontend/                   # Vite/TypeScript client
├── util/export_model.py        # .ckpt -> .pt export helper
├── train.sh                    # Quick pretrain script
└── sgf/                        # Optional SGF corpus
```

## Requirements

- Python 3.10+
- Node.js 18+

Install Python deps:

```bash
uv pip install -r requirements.txt
```

Install frontend deps:

```bash
cd frontend
npm install
```

## Training

Pretrain only:

```bash
python3 main.py --sgf-dir sgf --pretrain-epochs 10 --selfplay-iters 0
```

Pretrain + self-play:

```bash
python3 main.py --sgf-dir sgf --pretrain-epochs 10 --selfplay-iters 5
```

Useful flags:

- `--sgf-dir`: SGF input directory
- `--pretrain-epochs`: supervised epochs
- `--selfplay-iters`: RL iterations
- `--games-per-iter`: self-play games per iteration
- `--mcts-sims`: MCTS simulations per move
- `--batch-size`: training batch size
- `--output`: final model path (default `go_model.pt`)

## Exporting checkpoints

If you trained with Lightning checkpoints and want plain inference weights:

```bash
python3 util/export_model.py --checkpoint lightning_logs/<version>/checkpoints/<file>.ckpt --output go_model.pt
```

## Running the backend

The API server loads `MODEL_PATH` (default `./go_model.pt`).

```bash
MODEL_PATH=go_model_pretrain.pt python3 backend/server.py
```

Environment config:

- `PORT` (default `5000`)
- `MCTS_SIMULATIONS` (default `800`)
- `LOG_LEVEL` (default `INFO`)

API endpoints:

- `POST /api/game/new`
- `GET /api/game/{game_id}`
- `POST /api/game/move`

## Running the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`.

Build frontend static assets:

```bash
cd frontend
npm run build
```

Then start backend; it serves `frontend/dist` when present.


