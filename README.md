# Spectra

Spectra is an advanced video analysis tool that evaluates collision risks and temporal metrics using computer vision techniques.

## Features
- **Video Analysis**: Upload videos to process and analyze depth, motion, and segmentation data.
- **Risk Assessment**: Evaluates objects approaching the camera and calculates Time-to-Collision (TTC) and hazard scores.
- **Temporal Analysis**: Provides a detailed timeline of events mapping hazards across the video's duration.
- **Modern Interface**: A responsive, neural-core style dark interface for seamless user experience and clear data visualization.

## Installation

1. Ensure Python 3.8+ is installed.
2. Install the required dependencies:
   ```bash
   cd depth_project
   pip install -r requirements.txt
   ```

## Running the Application

You can easily start the application server using the provided `start.sh` script from the root directory:

```bash
./start.sh
```

To stop the server, use the `stop.sh` script:

```bash
./stop.sh
```

Once running, the application will be accessible via `http://127.0.0.1:8000`.

## Project Structure
- `depth_project/` - Main application folder containing the Flask backend and Python processors.
  - `static/` - Frontend assets (HTML, CSS, JS).
  - `app.py` - Main Flask application routing and logic.
  - `depth_estimator.py`, `motion_analyzer.py`, etc. - Computer vision processing modules.
