#!/usr/bin/env python3
import sys
import os
import torch
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.brain import ChessMimicNet
from scripts.preprocess import board_to_tensor, encode_move


def load_brain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessMimicNet().to(device)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model", "mimic_v1.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def get_mimic_move(model, device, board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None  # No moves available (Game Over)

    best_move = legal_moves[0]
    highest_prob = -1.0

    tensor = board_to_tensor(board)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    for move in legal_moves:
        move_idx = encode_move(move)
        prob = probabilities[move_idx].item()
        if prob > highest_prob:
            highest_prob = prob
            best_move = move

    return best_move


def main():
    model, device = load_brain()
    board = chess.Board()

    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        if line == "uci":
            print("id name Ritesh_Mimic_Bot")
            print("id author Ritesh")

            print("option name Move Overhead type spin default 500 min 0 max 5000")
            print("option name Threads type spin default 1 min 1 max 128")
            print("option name Hash type spin default 16 min 1 max 1024")

            print("uciok")
            sys.stdout.flush()

        elif line == "isready":
            print("readyok")
            sys.stdout.flush()

        elif line.startswith("position"):
            parts = line.split()
            if "startpos" in parts:
                board.reset()
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    for move_str in parts[moves_idx + 1 :]:
                        board.push(chess.Move.from_uci(move_str))

        elif line.startswith("go"):
            move = get_mimic_move(model, device, board)
            if move is not None:
                print(f"bestmove {move.uci()}")
            else:
                print("bestmove 0000")  # UCI standard signal for resignation / no moves
            sys.stdout.flush()

        elif line == "quit":
            break


if __name__ == "__main__":
    main()
