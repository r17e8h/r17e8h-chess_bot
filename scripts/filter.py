import chess.pgn
import os

RAW_PGN = "data/my_games.pgn"
FILTERED_PGN = "data/filtered_games.pgn"
MY_USERNAME = "r17e8h"
MIN_RATING = 1100
START_YEAR = "2025"


def filter_rapid_games():
    if not os.path.exists(RAW_PGN):
        print(f"Error: {RAW_PGN} not found. Run your bash download script first!")
        return

    input_pgn = open(RAW_PGN, "r")
    output_pgn = open(FILTERED_PGN, "w")

    total_processed = 0
    saved_count = 0

    print("Scrubbing data... Filtering for 10-minute Rapid games via TimeControl.")

    while True:
        game = chess.pgn.read_game(input_pgn)
        if game is None:
            break

        total_processed += 1

        time_control = game.headers.get("TimeControl", "")
        if time_control != "600":
            continue

        is_white = game.headers.get("White") == MY_USERNAME
        is_black = game.headers.get("Black") == MY_USERNAME

        if not (is_white or is_black):
            continue

        try:
            my_rating = int(
                game.headers.get("WhiteElo", "0")
                if is_white
                else game.headers.get("BlackElo", "0")
            )
        except (ValueError, TypeError):
            continue

        game_date = game.headers.get("Date", "0000.00.00")
        game_year = game_date.split(".")[0]

        if my_rating >= MIN_RATING and game_year >= START_YEAR:
            output_pgn.write(str(game) + "\n\n")
            saved_count += 1

    input_pgn.close()
    output_pgn.close()

    print(f"Processed {total_processed} games total.")
    print(f"Filtered down to {saved_count} pristine, high-quality Rapid games!")


if __name__ == "__main__":
    filter_rapid_games()
