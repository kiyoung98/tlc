#!/bin/bash

cd ../../

# Path to the directory
target_dir="log/alanine/300.0/1nano"

# Check if directory exists
if [ ! -d "$target_dir" ]; then
  echo "Error: Directory $target_dir does not exist"
  exit 1
fi

# List only directories in the specified path
echo "Directories in $target_dir:"
ls -d "$target_dir"/*/ 2>/dev/null || echo "No directories found"

# Loop through each directory and pass it to log.py
echo
echo "Running log.py for each directory in $target_dir"
for dir in "$target_dir"/*/; do
  echo "${dir%/}"
  if [ -d "$dir" ]; then
    python log.py --path "${dir%/}"
  fi
done