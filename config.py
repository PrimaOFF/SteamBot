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
    
    # Request configuration - Optimized for maximum throughput
    REQUEST_DELAY = 0.15  # Aggressive delay - Steam allows ~6-7 requests/second
    BURST_REQUEST_DELAY = 0.1  # For rapid bursts
    MAX_RETRIES = 5
    TIMEOUT = 15
    
    # Rate limiting and performance
    REQUESTS_PER_SECOND_LIMIT = 6  # Conservative estimate of Steam's limit
    ADAPTIVE_RATE_LIMITING = True  # Adjust delays based on response times
    CONNECTION_POOL_SIZE = 10  # Reuse connections for efficiency
    
    # Backoff configuration for rate limiting
    EXPONENTIAL_BACKOFF = {
        'initial_delay': 1.0,
        'max_delay': 60.0,
        'multiplier': 2.0,
        'max_attempts': 5
    }
    
    # Concurrent processing
    MAX_CONCURRENT_REQUESTS = 3  # Number of parallel requests
    MAX_WORKER_THREADS = 5  # For parallel processing
    
    # Scanning optimization
    AGGRESSIVE_SCANNING_MODE = True  # Enable maximum speed scanning
    INTELLIGENT_RETRY = True  # Smart retry logic for failed requests
    
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