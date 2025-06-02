"""
Configuration settings for the GeoNet Quake Classifier.
"""

# API Configuration
API_BASE_URL = "http://wfs.geonet.org.nz/geonet/ows"

# Base URL for individual earthquake pages on GeoNet
GEONET_EARTHQUAKE_URL_BASE = "https://www.geonet.org.nz/earthquake/"

# Filter Configuration
DAYS_FILTER = 7  # Number of days to look back for earthquakes

# Default filter parameters for WFS request
DEFAULT_FILTERS = {
    "service": "WFS",
    "version": "1.0.0",
    "request": "GetFeature",
    "typeName": "geonet:quake_search_v1",
    "outputFormat": "json",
    "maxFeatures": 10000,  # Increased to get more earthquakes
}

# Intensity classification threshold
# Earthquakes with MMI >= 4 are classified as "High" intensity
INTENSITY_THRESHOLD = 4

# Minimum magnitude filter (e.g., exclude small tremors)
MIN_MAGNITUDE = 3

# Map Configuration
# Determines the default intensity type displayed on the map
# Set to 'actual' or 'predicted'
DECISION_TREE_MAP_INTENSITY_TYPE = "predicted"
NEURAL_NETWORK_MAP_INTENSITY_TYPE = "predicted"

# Neural Network Configuration
NEURAL_NETWORK_CONFIG = {
    # Model Architecture
    "hidden_layers": [64, 32, 16],  # Number of units in each hidden layer
    "activation": "relu",  # Activation function for hidden layers
    "output_activation": "sigmoid",  # Activation function for output layer
    "dropout_rate": 0.2,  # Dropout rate for regularization
    "batch_norm": True,  # Whether to use batch normalization
    # Training Parameters
    "learning_rate": 0.001,  # Initial learning rate
    "batch_size": 32,  # Batch size for training
    "epochs": 100,  # Maximum number of epochs
    "validation_split": 0.2,  # Fraction of data to use for validation
    # Early Stopping
    "early_stopping": {
        "monitor": "val_loss",
        "patience": 10,
        "restore_best_weights": True,
    },
    # Learning Rate Reduction
    # fmt: off
    "reduce_lr": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6
    },
    # Self-calibration Settings
    "self_calibration": {
        "enabled": True,  # Whether to enable self-calibration
        "min_data_points": 200,  # Minimum data points required for training
        "max_data_points": 1000,  # Maximum data points to use for training
        "layer_size_multiplier": 2,  # Multiplier for layer sizes based on data
        "batch_size_multiplier": 4,  # Multiplier for batch size based on data
        "learning_rate_range": (1e-4, 1e-2),  # Range for
        # learning rate adjustment
        "epochs_range": (50, 200),  # Range for epochs adjustment
    },
    # fmt: on
}
