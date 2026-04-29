#!/bin/bash
# This script stops the running Spectra server.

echo "Stopping Spectra server..."

# Kill the uvicorn process that runs the Spectra app
if pkill -f "uvicorn zone_risk.app:app"; then
    echo "Server stopped successfully."
else
    echo "No running server found."
fi
