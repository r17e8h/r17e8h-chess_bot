import chess
import torch
from scripts.preprocess import board_to_tensor, encode_move

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
INNER_RING = {
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.F4,
    chess.C5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
}
SAFE_KING_SQUARES = {
    chess.G1,
    chess.H1,
    chess.C1,
    chess.B1,
    chess.A1,
    chess.G8,
    chess.H8,
    chess.C8,
    chess.B8,
    chess.A8,
}


def evaluate_board(board, plies_played=0):
    if board.is_checkmate():
        return -99999 + plies_played

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_draw()
    ):
        return 0

    score = 0
    if board.has_castling_rights(chess.WHITE):
        score += 30
    if board.has_castling_rights(chess.BLACK):
        score -= 30

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = PIECE_VALUES[piece.piece_type]

            if piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.PAWN]:
                if square in CENTER_SQUARES:
                    val += 30
                elif square in INNER_RING:
                    val += 10
            elif piece.piece_type == chess.KING:
                if square in SAFE_KING_SQUARES:
                    val += 50
                elif square in CENTER_SQUARES:
                    val -= 100
                elif square in INNER_RING:
                    val -= 50

            if piece.color == chess.WHITE:
                score += val
            else:
                score -= val

    return score if board.turn == chess.WHITE else -score


def get_move_score(board, move):
    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_val = (
            PIECE_VALUES[victim.piece_type] if victim else PIECE_VALUES[chess.PAWN]
        )
        attacker_val = (
            PIECE_VALUES[attacker.piece_type] if attacker else PIECE_VALUES[chess.PAWN]
        )
        score = 10 * victim_val - attacker_val + 10000
    if move.promotion:
        score += PIECE_VALUES[move.promotion] + 5000
    return score


def order_moves_for_alpha_beta(board, legal_moves):
    return sorted(legal_moves, key=lambda m: get_move_score(board, m), reverse=True)


def quiescence_search(board, alpha, beta, depth_limit=5):
    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
    if depth_limit == 0:
        return alpha

    capture_moves = [move for move in board.legal_moves if board.is_capture(move)]
    capture_moves = order_moves_for_alpha_beta(board, capture_moves)

    for move in capture_moves:
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, depth_limit - 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def get_model_candidates(board, legal_moves, model, device="cpu"):
    tensor = board_to_tensor(board)
    input_tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    move_scores = []
    for move in legal_moves:
        move_idx = encode_move(move)
        prob = probabilities[move_idx].item()
        move_scores.append((prob, move))

    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move for _, move in move_scores]


def negamax(board, depth, alpha, beta, plies_played=0):
    if board.is_game_over():
        return evaluate_board(board, plies_played)

    if depth == 0:
        return quiescence_search(board, alpha, beta)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return evaluate_board(board, plies_played)

    ordered_moves = order_moves_for_alpha_beta(board, legal_moves)

    max_eval = float("-inf")
    for move in ordered_moves:
        board.push(move)
        eval_score = -negamax(board, depth - 1, -beta, -alpha, plies_played + 1)
        board.pop()

        max_eval = max(max_eval, eval_score)
        alpha = max(alpha, eval_score)
        if alpha >= beta:
            break

    return max_eval


def select_best_move(board, depth, model, device):
    best_move = None
    max_eval = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    cnn_sorted_moves = get_model_candidates(board, legal_moves, model, device)

    captures = []
    quiets = []
    for move in cnn_sorted_moves:
        if board.is_capture(move) or move.promotion:
            captures.append(move)
        else:
            quiets.append(move)

    captures.sort(key=lambda m: get_move_score(board, m), reverse=True)

    root_candidates = captures + quiets

    for move in root_candidates:
        board.push(move)
        eval_score = -negamax(board, depth - 1, -beta, -alpha, 1)
        board.pop()

        if eval_score > max_eval:
            max_eval = eval_score
            best_move = move

        alpha = max(alpha, eval_score)

    return best_move if best_move else root_candidates[0]
