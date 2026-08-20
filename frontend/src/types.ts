export const BOARD_SIZE = 13;

export enum Stone {
  EMPTY = 0,
  BLACK = 1,
  WHITE = 2,
}

export interface Position {
  x: number;
  y: number;
}

export interface GameState {
  id: string;
  board: Stone[][];
  to_play: Stone;
  passes: number;
  ko: Position | null;
}

export interface MoveRequest {
  game_id: string;
  x: number;
  y: number;
  pass_move: boolean;
}

export interface MoveResponse {
  game: GameState;
  game_over: boolean;
  ai_move?: {
    x: number;
    y: number;
    pass: boolean;
  };
}

export type MessageType = 'info' | 'success' | 'error' | 'waiting';
