#!/bin/bash
# This script stops the running Spectrum server.

echo "Stopping Spectrum server..."

# Kill the uvicorn process that runs the Spectrum app
if pkill -f "uvicorn zone_risk.app:app"; then
    echo "Server stopped successfully."
else
    echo "No running server found."
fi
