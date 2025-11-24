#!/bin/bash

# Check if URL argument is provided
if [ -z "$1" ]; then
    echo "Error: URL or file path is required" >&2
    echo "Usage: $0 <URL_or_FILE_PATH>" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  $0 https://www.kantei.go.jp/jp/singi/example/" >&2
    echo "  $0 /path/to/document.pdf" >&2
    exit 1
fi

docker exec -it jpgovsummary-devcontainer bash -c "/workspaces/jpgovsummary/.venv/bin/python /workspaces/jpgovsummary/.venv/bin/jpgovsummary --batch $1"
