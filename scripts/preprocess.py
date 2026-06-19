import chess
import chess.pgn
import numpy as np
import os

FILTERED_PGN = "data/filtered_games.pgn"
OUTPUT_NPZ = "data/dataset.npz"
MY_USERNAME = "r17e8h"


def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        layer_idx = piece_map[piece.piece_type] + (
            6 if piece.color == chess.BLACK else 0
        )
        tensor[layer_idx, rank, file] = 1.0

    return tensor


def encode_move(move):
    return (move.from_square * 64) + move.to_square


def generate_dataset():
    if not os.path.exists(FILTERED_PGN):
        print(f"Error: {FILTERED_PGN} not found. Run filter.py first.")
        return

    pgn = open(FILTERED_PGN, "r")
    X = []
    Y = []

    games_count = 0
    print("Translating chess history into deep learning matrices...")

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break

        games_count += 1
        board = game.board()

        is_white = game.headers.get("White") == MY_USERNAME
        my_color = chess.WHITE if is_white else chess.BLACK

        for move in game.mainline_moves():
            if board.turn == my_color:
                X.append(board_to_tensor(board))
                Y.append(encode_move(move))

            board.push(move)

    pgn.close()

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)

    print(f"Packed matrix array shapes: Inputs {X.shape} | Labels {Y.shape}")

    np.savez_compressed(OUTPUT_NPZ, x=X, y=Y)
    print(f"Preprocessing complete! Binary dataset saved to {OUTPUT_NPZ}")


if __name__ == "__main__":
    generate_dataset()
