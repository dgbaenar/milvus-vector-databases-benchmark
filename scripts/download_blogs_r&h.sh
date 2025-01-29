#!/bin/bash

# Output directory
EFS_DIR=data

# Create folder if not exists
mkdir -p "$EFS_DIR"

echo "Downloading blogs in: $EFS_DIR"

wget -e robots=off \
  --recursive \
  --no-clobber \
  --page-requisites \
  --html-extension \
  --convert-links \
  --restrict-file-names=windows \
  --domains=www.robinsonandhenry.com \
  --no-parent \
  --accept=html \
  -P "$EFS_DIR" \
  https://www.robinsonandhenry.com/blog/

echo "Dowload complete"
