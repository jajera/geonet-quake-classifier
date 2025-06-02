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

# Map Configuration
# Determines the default intensity type displayed on the map
# Set to 'actual' or 'predicted'
MAP_INTENSITY_TYPE = "predicted"

# Minimum magnitude filter (e.g., exclude small tremors)
MIN_MAGNITUDE = 3
