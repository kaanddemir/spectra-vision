#!/bin/bash
# This script stops the running Spectra server.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Stopping Spectra server..."

# Kill the uvicorn process that runs the Spectra app
if pkill -f "uvicorn spectra.app:app"; then
    echo "Server stopped successfully."
else
    echo "No running server found."
fi
