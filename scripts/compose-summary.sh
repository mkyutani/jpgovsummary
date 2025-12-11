#!/bin/bash

# Check if at least one URL argument is provided
if [ $# -eq 0 ]; then
    echo "Error: At least one URL or file path is required" >&2
    echo "Usage: $0 <URL_or_FILE_PATH> [URL_or_FILE_PATH...]" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 https://www.kantei.go.jp/jp/singi/example/" >&2
    echo "  $0 /path/to/document.pdf" >&2
    echo "  $0 https://example1.go.jp/ https://example2.go.jp/ /path/to/doc.pdf" >&2
    exit 1
fi

# Ensure container is running
docker compose up -d

# Process each URL/file path in order
for target in "$@"; do
    echo "Processing: $target" >&2
    docker compose exec jpgovsummary jpgovsummary --batch "$target"

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error: Failed to process $target" >&2
    fi
    echo "" >&2
done
