#!/bin/bash

# Check if a folder path is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder-path>"
    exit 1
fi

# assign the provided folder path to a variable
folder_path=$1

# Check if the provided path is a directory
if [ ! -d "$folder_path" ]; then
    echo "Error: The provided path is not a directory."
    exit 1
fi

# Find and print absolute paths of all folders named "000" within the specified folder
find "$folder_path" -type d -name "000" -exec sh -c 'echo "{}"; python3 extract_blinks.py "{}" && python3 extract_diameter.py "{}"' \;