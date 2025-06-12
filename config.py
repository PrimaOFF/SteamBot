import os
from typing import Dict, List, Tuple

class FloatCheckerConfig:
    # Steam API Configuration
    STEAM_API_KEY = os.getenv('STEAM_API_KEY', '')
    CS2_APP_ID = 730
    
    # Market URLs
    STEAM_MARKET_BASE_URL = "https://steamcommunity.com/market"
    STEAM_MARKET_LISTINGS_URL = f"{STEAM_MARKET_BASE_URL}/listings/{CS2_APP_ID}"
    STEAM_MARKET_SEARCH_URL = f"{STEAM_MARKET_BASE_URL}/search/render/"
    
    # Third-party APIs for float checking
    CSFLOAT_API_URL = "https://api.csfloat.com"
    TRADEIT_API_URL = "https://tradeit.gg/api"
    
    # Request configuration
    REQUEST_DELAY = 1.0  # Seconds between requests to avoid rate limiting
    MAX_RETRIES = 3
    TIMEOUT = 30
    
    # Float value thresholds for rarity detection
    RARE_FLOAT_THRESHOLDS = {
        'factory_new': {
            'min': 0.00,
            'max': 0.005,  # Very low float FN items
            'extreme_max': 0.001  # Extremely rare low floats
        },
        'battle_scarred': {
            'min': 0.995,  # Very high float BS items
            'max': 1.0,
            'extreme_min': 0.999  # Extremely rare high floats
        }
    }
    
    # Standard wear ranges
    WEAR_RANGES = {
        'Factory New': (0.00, 0.07),
        'Minimal Wear': (0.07, 0.15),
        'Field-Tested': (0.15, 0.37),
        'Well-Worn': (0.37, 0.45),
        'Battle-Scarred': (0.45, 1.00)
    }
    
    # Items to monitor (can be expanded)
    MONITORED_ITEMS = [
        "AK-47 | Redline",
        "AWP | Dragon Lore",
        "M4A4 | Howl",
        "Glock-18 | Fade",
        "Karambit | Doppler",
        "AK-47 | Fire Serpent",
        "M4A1-S | Hot Rod"
    ]
    
    # Database configuration
    DATABASE_PATH = "float_checker.db"
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FILE = "float_checker.log"
    
    # Notification settings
    ENABLE_NOTIFICATIONS = True
    NOTIFICATION_THRESHOLD_VALUE = 1000.0  # USD value threshold for notifications
    
    # Skin-specific float ranges (some skins have restricted ranges)
    SKIN_SPECIFIC_RANGES = {
        "M4A4 | Howl": {
            'Factory New': (0.00, 0.04),
            'Minimal Wear': (0.04, 0.08),
            'Field-Tested': (0.08, 0.37),
            'Well-Worn': (0.37, 0.45),
            'Battle-Scarred': (0.45, 1.00)
        },
        "AWP | Dragon Lore": {
            'Factory New': (0.00, 0.07),
            'Minimal Wear': (0.07, 0.15),
            'Field-Tested': (0.15, 0.37),
            'Well-Worn': (0.37, 0.45),
            'Battle-Scarred': (0.45, 1.00)
        }
        # Add more skin-specific ranges as needed
    }