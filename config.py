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
    
    # CSFloat API Configuration
    CSFLOAT_RATE_LIMIT = 10  # requests per minute
    CSFLOAT_MIN_INTERVAL = 6  # seconds between requests
    CSFLOAT_MAX_RETRIES = 3
    CSFLOAT_TIMEOUT = 30  # seconds
    CSFLOAT_CIRCUIT_BREAKER_THRESHOLD = 5  # failures before circuit breaker opens
    CSFLOAT_CACHE_DURATION_HOURS = 1  # hours to cache results
    
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
    
    # Comprehensive skin-specific float ranges and extreme thresholds
    SKIN_SPECIFIC_RANGES = {
        # AK-47 Series
        "AK-47 | Crossfade": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 0.50),  # Max 0.50
            'extreme_fn': 0.0001,  # Extremely rare FN
            'extreme_bs': 0.499    # Extremely rare BS (close to max)
        },
        "AK-47 | Redline": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        "AK-47 | Fire Serpent": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        "AK-47 | Case Hardened": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        "AK-47 | Vulcan": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        
        # AWP Series  
        "AWP | Dragon Lore": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        "AWP | Asiimov": {
            'Factory New': None,  # Doesn't exist in FN
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': None,
            'extreme_bs': 0.999
        },
        "AWP | Lightning Strike": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': None,  # Doesn't exist in BS
            'extreme_fn': 0.0001,
            'extreme_bs': None
        },
        "AWP | Hyper Beast": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        
        # M4A4 Series
        "M4A4 | Howl": {
            'Factory New': (0.00, 0.04),  # Restricted range
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        "M4A4 | Asiimov": {
            'Factory New': None,  # Doesn't exist in FN
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': None,
            'extreme_bs': 0.999
        },
        "M4A4 | Dragon King": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        
        # M4A1-S Series
        "M4A1-S | Hot Rod": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': None,  # Doesn't exist in BS
            'extreme_fn': 0.0001,
            'extreme_bs': None
        },
        "M4A1-S | Hyper Beast": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        "M4A1-S | Cyrex": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        
        # Glock Series
        "Glock-18 | Fade": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': None,  # Doesn't exist in BS
            'extreme_fn': 0.0001,
            'extreme_bs': None
        },
        "Glock-18 | Water Elemental": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        
        # USP-S Series
        "USP-S | Kill Confirmed": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        "USP-S | Orion": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        
        # Knife Series (most knives don't have BS)
        "Karambit | Doppler": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': None,
            'extreme_fn': 0.0001,
            'extreme_bs': None
        },
        "Karambit | Fade": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': None,
            'extreme_fn': 0.0001,
            'extreme_bs': None
        },
        "Butterfly Knife | Fade": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': None,
            'extreme_fn': 0.0001,
            'extreme_bs': None
        },
        
        # Gloves (special case - very limited wear ranges)
        "Sport Gloves | Pandora's Box": {
            'Factory New': None,  # Doesn't exist in FN
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': None,
            'extreme_bs': 0.999
        },
        "Driver Gloves | King Snake": {
            'Factory New': None,
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': None,
            'extreme_bs': 0.999
        },
        
        # Popular rifles with unique ranges
        "FAMAS | Afterimage": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 0.65),  # Limited BS range
            'extreme_fn': 0.0001,
            'extreme_bs': 0.649
        },
        "Galil AR | Cerberus": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': 0.0001,
            'extreme_bs': 0.999
        },
        
        # SMGs with special ranges
        "P90 | Asiimov": {
            'Factory New': None,
            'Battle-Scarred': (0.45, 1.00),
            'extreme_fn': None,
            'extreme_bs': 0.999
        },
        "MP7 | Skulls": {
            'Factory New': (0.00, 0.07),
            'Battle-Scarred': (0.45, 0.75),  # Limited range
            'extreme_fn': 0.0001,
            'extreme_bs': 0.749
        }
    }
    
    # Items that only exist in specific wear conditions
    WEAR_RESTRICTIONS = {
        # Items that don't exist in Factory New
        'no_factory_new': [
            "AWP | Asiimov",
            "M4A4 | Asiimov", 
            "P90 | Asiimov",
            "Sport Gloves | Pandora's Box",
            "Driver Gloves | King Snake",
            "Hand Wraps | Cobalt Skulls"
        ],
        
        # Items that don't exist in Battle-Scarred
        'no_battle_scarred': [
            "AWP | Lightning Strike",
            "M4A1-S | Hot Rod",
            "Glock-18 | Fade",
            "Karambit | Doppler",
            "Karambit | Fade", 
            "Butterfly Knife | Fade",
            "Bayonet | Doppler",
            "USP-S | Neo-Noir"
        ]
    }