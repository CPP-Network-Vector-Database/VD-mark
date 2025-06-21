#!/bin/bash

# List of directories to be deleted
directories="chroma milvus pgvector qdrant redis weaviate"

# Function to delete directories
delete_directories() {
    for dir in $directories; do
        if [ -d "$dir" ]; then
            echo "Deleting directory: $dir"
            sudo rm -rf "$dir"
        else
            echo "Directory '$dir' does not exist, skipping."
        fi
    done
}

# Start the deletion process
echo "This script will delete the following directories: $directories"
echo "Are you sure you want to proceed with deleting these directories? (y/n)"
read -r confirmation

if [ "$confirmation" = "y" ] || [ "$confirmation" = "Y" ]; then
    delete_directories
    echo "Deletion process complete."
else
    echo "Aborted. No directories were deleted."
fi