import { BOARD_SIZE, Stone, GameState, MoveRequest, MoveResponse, MessageType } from './types';

const $ = (selector: string): HTMLElement => {
  const element = document.getElementById(selector);
  if (!element) {
    throw new Error(`Element with id "${selector}" not found`);
  }
  return element;
};

export class GoBoard {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private gameId: string | null = null;
  private gameState: GameState | null = null;
  private isWaiting = false;
  private hoverX: number | null = null;
  private hoverY: number | null = null;

  private readonly cellSize = 50;
  private readonly margin = 25;
  private readonly stoneRadius = 20;

  constructor() {
    this.canvas = $('board') as HTMLCanvasElement;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }
    this.ctx = ctx;

    this.canvas.addEventListener('click', (e) => this.handleClick(e));
    this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    this.canvas.addEventListener('mouseleave', () => this.handleMouseLeave());
    $('new-game-btn').addEventListener('click', () => this.newGame());
    $('pass-btn').addEventListener('click', () => this.pass());

    this.newGame();
  }

  async newGame(): Promise<void> {
    this.showMessage('Starting new game...', 'info');
    try {
      const response = await fetch('/api/game/new', {
        method: 'POST',
      });
      const game: GameState = await response.json();
      this.gameId = game.id;
      this.gameState = game;
      this.draw();
      this.updateStatus();
      this.showMessage('New game started! You play as Black.', 'success');
    } catch (error) {
      this.showMessage(`Error starting game: ${error}`, 'error');
    }
  }

  async makeMove(x: number, y: number, pass = false): Promise<void> {
    if (this.isWaiting || !this.gameId || !this.gameState) return;

    this.isWaiting = true;
    let gameEnded = false;
    this.showMessage('Waiting for AI...', 'waiting');
    ($('pass-btn') as HTMLButtonElement).disabled = true;

    const previousState = JSON.parse(JSON.stringify(this.gameState)) as GameState;
    if (!pass) {
      this.gameState.board[y][x] = Stone.BLACK;
      this.draw();
    }

    try {
      const body: MoveRequest = {
        game_id: this.gameId,
        x,
        y,
        pass_move: pass,
      };

      const response = await fetch('/api/game/move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const error = await response.text();
        this.gameState = previousState;
        this.draw();
        throw new Error(error);
      }

      const data: MoveResponse = await response.json();
      this.gameState = data.game;
      this.draw();
      this.updateStatus();

      if (data.game_over) {
        this.showMessage('Game Over! Both players passed.', 'info');
        gameEnded = true;
      } else if (data.ai_move) {
        const aiMove = data.ai_move;
        if (aiMove.pass) {
          this.showMessage('AI passed', 'info');
        } else {
          this.showMessage(`AI played at (${aiMove.x}, ${aiMove.y})`, 'info');
        }
      }
    } catch (error) {
      this.showMessage(`Error: ${(error as Error).message}`, 'error');
    } finally {
      this.isWaiting = gameEnded;
      ($('pass-btn') as HTMLButtonElement).disabled = gameEnded;
    }
  }

  async pass(): Promise<void> {
    await this.makeMove(0, 0, true);
  }

  private handleClick(event: MouseEvent): void {
    if (this.isWaiting || !this.gameState || this.gameState.to_play !== Stone.BLACK) {
      return;
    }

    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const boardX = Math.round((x - this.margin) / this.cellSize);
    const boardY = Math.round((y - this.margin) / this.cellSize);

    if (boardX >= 0 && boardX < BOARD_SIZE && boardY >= 0 && boardY < BOARD_SIZE) {
      if (this.gameState.board[boardY][boardX] === Stone.EMPTY) {
        this.makeMove(boardX, boardY);
      }
    }
  }

  private handleMouseMove(event: MouseEvent): void {
    if (!this.gameState || this.gameState.to_play !== Stone.BLACK) {
      this.hoverX = null;
      this.hoverY = null;
      return;
    }

    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const boardX = Math.round((x - this.margin) / this.cellSize);
    const boardY = Math.round((y - this.margin) / this.cellSize);

    if (boardX >= 0 && boardX < BOARD_SIZE && boardY >= 0 && boardY < BOARD_SIZE) {
      if (this.gameState.board[boardY][boardX] === Stone.EMPTY) {
        this.hoverX = boardX;
        this.hoverY = boardY;
        this.draw();
        return;
      }
    }

    this.hoverX = null;
    this.hoverY = null;
    this.draw();
  }

  private handleMouseLeave(): void {
    this.hoverX = null;
    this.hoverY = null;
    this.draw();
  }

  private draw(): void {
    if (!this.gameState) return;

    const ctx = this.ctx;
    const { cellSize, margin } = this;

    ctx.fillStyle = '#dcb35c';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;

    for (let i = 0; i < BOARD_SIZE; i++) {
      ctx.beginPath();
      ctx.moveTo(margin + i * cellSize, margin);
      ctx.lineTo(margin + i * cellSize, margin + (BOARD_SIZE - 1) * cellSize);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(margin, margin + i * cellSize);
      ctx.lineTo(margin + (BOARD_SIZE - 1) * cellSize, margin + i * cellSize);
      ctx.stroke();
    }

    const starPoints: [number, number][] = [[3, 3], [3, 9], [9, 3], [9, 9], [6, 6]];
    ctx.fillStyle = '#000';
    starPoints.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(margin + x * cellSize, margin + y * cellSize, 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    for (let y = 0; y < BOARD_SIZE; y++) {
      for (let x = 0; x < BOARD_SIZE; x++) {
        const stone = this.gameState.board[y][x];
        if (stone !== Stone.EMPTY) {
          const cx = margin + x * cellSize;
          const cy = margin + y * cellSize;

          ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
          ctx.beginPath();
          ctx.arc(cx + 2, cy + 2, this.stoneRadius, 0, 2 * Math.PI);
          ctx.fill();

          if (stone === Stone.BLACK) {
            const gradient = ctx.createRadialGradient(
              cx - 5, cy - 5, 2,
              cx, cy, this.stoneRadius
            );
            gradient.addColorStop(0, '#666');
            gradient.addColorStop(1, '#000');
            ctx.fillStyle = gradient;
          } else {
            const gradient = ctx.createRadialGradient(
              cx - 5, cy - 5, 2,
              cx, cy, this.stoneRadius
            );
            gradient.addColorStop(0, '#fff');
            gradient.addColorStop(1, '#ddd');
            ctx.fillStyle = gradient;
          }

          ctx.beginPath();
          ctx.arc(cx, cy, this.stoneRadius, 0, 2 * Math.PI);
          ctx.fill();

          ctx.strokeStyle = stone === Stone.BLACK ? '#000' : '#aaa';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    if (this.gameState.ko) {
      const ko = this.gameState.ko;
      const cx = margin + ko.x * cellSize;
      const cy = margin + ko.y * cellSize;

      ctx.strokeStyle = '#f00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, this.stoneRadius + 5, 0, 2 * Math.PI);
      ctx.stroke();
    }

    if (this.hoverX !== null && this.hoverY !== null && !this.isWaiting) {
      const cx = margin + this.hoverX * cellSize;
      const cy = margin + this.hoverY * cellSize;

      const currentPlayer = this.gameState.to_play;
      if (currentPlayer === Stone.BLACK) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      } else {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      }

      ctx.beginPath();
      ctx.arc(cx, cy, this.stoneRadius, 0, 2 * Math.PI);
      ctx.fill();

      ctx.strokeStyle = currentPlayer === Stone.BLACK ? 'rgba(0, 0, 0, 0.5)' : 'rgba(170, 170, 170, 0.7)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  private updateStatus(): void {
    if (!this.gameState) return;

    const turnIndicator = $('turn-indicator');
    const passesElement = $('passes');

    const colorName = this.gameState.to_play === Stone.BLACK ? 'Black' : 'White';
    const playerInfo = this.gameState.to_play === Stone.BLACK ? 'Your turn' : 'AI thinking...';

    turnIndicator.textContent = `${colorName} to play (${playerInfo})`;
    passesElement.textContent = `Passes: ${this.gameState.passes}`;
  }

  private showMessage(text: string, type: MessageType = 'info'): void {
    const messageDiv = $('message');
    messageDiv.textContent = text;
    messageDiv.className = type;
  }
}
