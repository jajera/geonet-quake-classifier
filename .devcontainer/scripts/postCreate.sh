#!/bin/bash

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (linters, etc.)
pip install -r requirements-dev.txt

# Run linters to check code style and quality
echo "Running flake8..."
flake8 .

echo "Running pylint..."
pylint decision_tree.py neural_network.py

# Execute the scripts to generate output files
echo "Executing decision_tree.py..."
python decision_tree.py

echo "Executing neural_networks.py..."
python neural_network.py

echo "postCreate script finished."
