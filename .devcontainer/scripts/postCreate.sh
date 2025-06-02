#!/bin/bash

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (linters, etc.)
pip install -r requirements-dev.txt

# Run linters to check code style and quality
echo "Running flake8..."
flake8 .

echo "Running pylint..."
pylint quake_classifier.py

# Execute the main script to generate output files (e.g., map HTML)
echo "Executing quake_classifier.py..."
python quake_classifier.py

echo "postCreate script finished."
