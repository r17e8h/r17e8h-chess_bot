#!/bin/bash

USERNAME="r17e8h"
OUT_FILE="../data/my_games.pgn"
USER_AGENT="Chess-Mimic-Bot-Project (Contact: r17e8h@proton.me)"

mkdir -p ../data

echo "Starting game harvest for $USERNAME..."

ARCHIVES=$(curl -s -H "User-Agent: $USER_AGENT" \
  "https://api.chess.com/pub/player/$USERNAME/games/archives" | jq -r '.archives[]')

>"$OUT_FILE"

for url in $ARCHIVES; do
  DATE_LABEL=$(echo "$url" | grep -oE '[0-9]{4}/[0-9]{2}')
  echo "Downloading: $DATE_LABEL..."
  curl -s -H "User-Agent: $USER_AGENT" "$url/pgn" >>"$OUT_FILE"
  echo -e "\n\n" >>"$OUT_FILE"
done

echo "Success! Every single game downloaded to $OUT_FILE"
