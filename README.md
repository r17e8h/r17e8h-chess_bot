# r17e8h-chess_bot
Okay so, I’ve been playing a lot of chess, and I thought—why not build a bot that actually plays like me? Not some super-engine like Stockfish that plays perfectly, but something that makes the same weird moves (and the occasional blunders) that I do.

I’m using this project to dive deep into a few things I’ve been wanting to learn:
- **ML & Behavioral Cloning:** Instead of teaching the bot to play "good" chess, I'm training a PyTorch model to predict what move I would make in any given position.
- **Data Pipelines:** Writing Bash scripts to pull my own game history from chess.com and using python-chess to turn that PGN data into tensors my model can actually understand.
- **API Integration:** Getting this whole thing live by connecting it to the Lichess Bot API so I can eventually use this in my future projects. idk i guess a portfolio :P
