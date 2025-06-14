#!/bin/bash

# postCreate.sh - Development container post-creation script
# This script runs after the development container is created

echo "🚀 Setting up GeoNet Earthquake Classifier Development Environment"
echo "================================================================="

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install pandas scikit-learn matplotlib numpy requests urllib3

# Optional: Install TensorFlow for the original neural network model
echo "🧠 Installing TensorFlow (optional)..."
pip3 install tensorflow

# Optional: Install HuggingFace Transformers for transformer model
echo "🤖 Installing HuggingFace Transformers (optional)..."
pip3 install transformers torch

# Install development tools
echo "🔧 Installing development tools..."
pip3 install pylint black

# Make scripts executable
echo "🔑 Setting script permissions..."
chmod +x *.py

# Run linting on all model files
echo "🔍 Running pylint checks..."
echo "Linting utilities.py..."
python3 -m pylint utilities.py || true
echo "Linting decision_tree.py..."
python3 -m pylint decision_tree.py || true
echo "Linting neural_network.py..."
python3 -m pylint neural_network.py || true
echo "Linting statistical_model.py..."
python3 -m pylint statistical_model.py || true
echo "Linting ml_model.py..."
python3 -m pylint ml_model.py || true
echo "Linting neural_model.py..."
python3 -m pylint neural_model.py || true
echo "Linting transformer_model.py..."
python3 -m pylint transformer_model.py || true

# Test run all models to ensure they work
echo "🧪 Testing all earthquake classifier models..."

echo "Running Decision Tree model..."
python3 decision_tree.py || echo "❌ Decision Tree model failed"

echo "Running Neural Network model (TensorFlow)..."
python3 neural_network.py || echo "❌ Neural Network model failed"

echo "Running Statistical model..."
python3 statistical_model.py || echo "❌ Statistical model failed"

echo "Running ML model..."
python3 ml_model.py || echo "❌ ML model failed"

echo "Running Neural model (MLPClassifier)..."
python3 neural_model.py || echo "❌ Neural model failed"

echo "Running Transformer model (DistilBERT)..."
python3 transformer_model.py || echo "❌ Transformer model failed"

# List generated files
echo "📁 Generated files:"
ls -la *.html *.png 2>/dev/null || echo "No HTML/PNG files generated yet"

echo "✅ Post-creation setup complete!"
echo "🗺️  Open index.html to view the project dashboard"
echo "📊 Individual model results:"
echo "   - decision_tree.html (Decision Tree)"
echo "   - neural_network.html (Neural Network - TensorFlow)"
echo "   - statistical_model.html (Statistical Model)"
echo "   - ml_model.html (Machine Learning Model)"
echo "   - neural_model.html (Neural Model - MLPClassifier)"
echo "   - transformer_model.html (Transformer Model - DistilBERT)"
