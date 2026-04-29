#!/bin/bash
# This script automatically activates the virtual environment and starts the application.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv directory not found! Please set up the virtual environment first."
    exit 1
fi

echo "========================================"
echo "Starting Spectra Server..."
echo "========================================"

# Set PYTHONPATH to include the base directory so realtime_danger can be imported
export PYTHONPATH="${DIR}:${PYTHONPATH}"

# Change to depth_project directory and start uvicorn
cd "${DIR}/depth_project"
echo "Server will be available at: http://localhost:8000"
python -m uvicorn app:app --host localhost --port 8000 --reload
