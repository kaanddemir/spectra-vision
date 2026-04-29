#!/bin/bash
# This script stops the running Spectra server.

echo "Stopping Spectra server..."

# Kill the uvicorn process that runs app:app
if pkill -f "uvicorn app:app"; then
    echo "Server stopped successfully."
else
    echo "No running server found."
fi
