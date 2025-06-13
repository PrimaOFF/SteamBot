import requests
import json
import time
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import logging
from config import FloatCheckerConfig

class SteamMarketAPI:
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.logger = logging.getLogger(__name__)
    
    def search_items(self, search_term: str, start: int = 0, count: int = 100) -> Dict:
        """Search for items on Steam Market"""
        params = {
            'query': search_term,
            'start': start,
            'count': count,
            'search_descriptions': 0,
            'sort_column': 'popular',
            'sort_dir': 'desc',
            'appid': self.config.CS2_APP_ID,
            'category_730_ItemSet[]': 'any',
            'category_730_ProPlayer[]': 'any',
            'category_730_StickerCapsule[]': 'any',
            'category_730_TournamentTeam[]': 'any',
            'category_730_Weapon[]': 'any'
        }
        
        try:
            response = self.session.get(
                self.config.STEAM_MARKET_SEARCH_URL,
                params=params,
                timeout=self.config.TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error searching items: {e}")
            return {}
    
    def get_item_listings(self, market_hash_name: str, start: int = 0, count: int = 100) -> Dict:
        """Get listings for a specific item"""
        encoded_name = quote(market_hash_name)
        url = f"{self.config.STEAM_MARKET_LISTINGS_URL}/{encoded_name}/render/"
        
        params = {
            'query': '',
            'start': start,
            'count': count,
            'country': 'US',
            'language': 'english',
            'currency': 1
        }
        
        try:
            response = self.session.get(url, params=params, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error getting listings for {market_hash_name}: {e}")
            return {}
    
    def get_price_history(self, market_hash_name: str) -> Dict:
        """Get price history for an item"""
        url = f"{self.config.STEAM_MARKET_BASE_URL}/pricehistory/"
        
        params = {
            'appid': self.config.CS2_APP_ID,
            'market_hash_name': market_hash_name
        }
        
        try:
            response = self.session.get(url, params=params, timeout=self.config.TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error getting price history for {market_hash_name}: {e}")
            return {}
    
    def extract_inspect_links(self, listings_data: Dict) -> List[str]:
        """Extract inspect links from listings data"""
        inspect_links = []
        
        if 'listinginfo' in listings_data:
            for listing_id, listing_info in listings_data['listinginfo'].items():
                if 'asset' in listing_info:
                    asset = listing_info['asset']
                    if 'market_actions' in asset:
                        for action in asset['market_actions']:
                            if 'link' in action and 'inspect' in action['link'].lower():
                                inspect_links.append(action['link'])
        
        return inspect_links
    
    def parse_inspect_link(self, inspect_link: str) -> Optional[Dict]:
        """Parse inspect link to extract useful information"""
        # Extract parameters from inspect link
        pattern = r'steam://rungame/730/76561202255233023/\+csgo_econ_action_preview%20([^%]+)'
        match = re.search(pattern, inspect_link)
        
        if match:
            params_str = match.group(1)
            # Parse parameters like S76561198084749846A123456789D456789123456789
            param_pattern = r'S(\d+)A(\d+)D(\d+)'
            param_match = re.search(param_pattern, params_str)
            
            if param_match:
                return {
                    'steamid': param_match.group(1),
                    'assetid': param_match.group(2),
                    'classid': param_match.group(3),
                    'full_link': inspect_link
                }
        
        return None
    
    def get_item_float_via_third_party(self, inspect_link: str) -> Optional[float]:
        """Get float value using third-party API (CSFloat)"""
        # This would require implementing the third-party API call
        # For now, return None as placeholder
        # In a real implementation, you'd send the inspect link to CSFloat API
        self.logger.info(f"Would check float for inspect link: {inspect_link}")
        return None
    
    def rate_limit_delay(self):
        """Apply rate limiting delay"""
        time.sleep(self.config.REQUEST_DELAY)
    
    def get_extreme_float_variants(self, base_skin_name: str) -> List[str]:
        """Get only Factory New and Battle-Scarred variants for extreme float scanning"""
        variants = []
        
        # Check if skin has specific restrictions
        if base_skin_name in self.config.WEAR_RESTRICTIONS['no_factory_new']:
            # Only add Battle-Scarred if it exists
            if base_skin_name not in self.config.WEAR_RESTRICTIONS['no_battle_scarred']:
                variants.append(f"{base_skin_name} (Battle-Scarred)")
        elif base_skin_name in self.config.WEAR_RESTRICTIONS['no_battle_scarred']:
            # Only add Factory New
            variants.append(f"{base_skin_name} (Factory New)")
        else:
            # Add both FN and BS if they exist
            variants.append(f"{base_skin_name} (Factory New)")
            variants.append(f"{base_skin_name} (Battle-Scarred)")
        
        return variants
    
    def get_all_skin_variants(self, base_skin_name: str) -> List[str]:
        """Get all wear variants of a skin (legacy method)"""
        variants = []
        for wear in self.config.WEAR_RANGES.keys():
            variant_name = f"{base_skin_name} ({wear})"
            variants.append(variant_name)
        return variants
    
    def is_extreme_float_candidate(self, float_value: float, item_name: str) -> bool:
        """Check if a float value qualifies as extreme for the specific item"""
        # Extract base skin name
        base_skin = item_name.split(' (')[0] if ' (' in item_name else item_name
        
        # Get skin-specific thresholds
        if base_skin in self.config.SKIN_SPECIFIC_RANGES:
            skin_data = self.config.SKIN_SPECIFIC_RANGES[base_skin]
            
            # Check Factory New extreme
            if 'Factory New' in item_name:
                extreme_fn = skin_data.get('extreme_fn')
                return extreme_fn is not None and float_value <= extreme_fn
            
            # Check Battle-Scarred extreme  
            elif 'Battle-Scarred' in item_name:
                extreme_bs = skin_data.get('extreme_bs')
                return extreme_bs is not None and float_value >= extreme_bs
        
        # Fallback to generic thresholds
        if 'Factory New' in item_name:
            return float_value <= 0.0001
        elif 'Battle-Scarred' in item_name:
            return float_value >= 0.999
        
        return False