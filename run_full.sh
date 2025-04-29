#!/bin/bash

rm -rf infractions*

# Start the Flask app in the background and save its PID
python3 gui/app.py &
FLASK_PID=$!

# Start the pipeline in the background and save its PID
python3 pipeline/full_pipeline.py &
PIPELINE_PID=$!

# Wait for processes or until you decide to kill them
echo "Flask PID: $FLASK_PID"
echo "Pipeline PID: $PIPELINE_PID"

# Press Ctrl+C to exit, then cleanup:
trap "echo 'Killing background processes'; kill $FLASK_PID $PIPELINE_PID; exit 0" SIGINT SIGTERM

wait