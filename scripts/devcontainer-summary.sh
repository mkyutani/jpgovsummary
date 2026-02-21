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

# Container name defined in .devcontainer/devcontainer.json runArgs
CONTAINER_NAME="jpgovsummary-devcontainer"

# Ensure container is running
if docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
    : # already running
elif docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "Starting devcontainer..." >&2
    docker start "$CONTAINER_NAME" >/dev/null
else
    echo "Error: Devcontainer '$CONTAINER_NAME' not found. Create it from VSCode first." >&2
    exit 1
fi

# Process each URL/file path in order
for target in "$@"; do
    echo "Processing: $target" >&2
    docker exec "$CONTAINER_NAME" bash -l -c "cd /workspaces/jpgovsummary && poetry run jpgovsummary --batch $JPGOVSUMMARY_OPTIONS '$target'"

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error: Failed to process $target" >&2
    fi
    echo "" >&2
done
