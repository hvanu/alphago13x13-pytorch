#!/bin/bash

# Download SGF files as P0.sgf..P100.sgf from a common base URL.
# Example:
#   SGF_BASE_URL="https://example.org/games" ./download_sgf.sh

mkdir -p sgf

if [ -z "$SGF_BASE_URL" ]; then
  echo "Set SGF_BASE_URL to the directory hosting P*.sgf files"
  exit 1
fi

for i in {0..100}; do
  url="$SGF_BASE_URL/P${i}.sgf"
  dest="sgf/P${i}.sgf"
  curl -f -o "$dest" "$url" || echo "File P${i}.sgf not found, skipping."
done