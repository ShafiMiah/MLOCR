#!/bin/bash

# List of commands
commands=(
    "pyinstaller --onefile --noconsole --clean Main_ODAI.py"
)

# Define folders to copy (source -> target)
folders_to_copy=(
    "Config:dist"
    "runs:dist"
    "yolov8models:dist"
)

# Define files to copy (source -> target)
files_to_copy=(
    "Train.pt:dist"
    "yolov8s.pt:dist"
)

# Run each command
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"

    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd"
        exit 1
    fi

    # Copy folders
    for pair in "${folders_to_copy[@]}"; do
        IFS=":" read -r source target <<< "$pair"
        if [ -d "$source" ]; then
            mkdir -p "$target"
            cp -r "$source" "$target"
            echo "Copied $source to $target"
        else
            echo "Source folder $source does not exist, skipping copy."
        fi
    done

    # Copy files
    for pair in "${files_to_copy[@]}"; do
        IFS=":" read -r source target <<< "$pair"
        if [ -f "$source" ]; then
            mkdir -p "$target"
            cp "$source" "$target"
            echo "Copied $source to $target"
        else
            echo "Source file $source does not exist, skipping copy."
        fi
    done

    echo "--------------------"
done