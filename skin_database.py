import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class SkinInfo:
    name: str
    weapon: str
    pattern: str
    rarity: str
    min_float: float
    max_float: float
    wear_ranges: Dict[str, Tuple[float, float]]
    stattrak: bool = False
    souvenir: bool = False

class SkinDatabase:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.skins_data = {}
        self.weapons_list = []
        self.api_base_url = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en"
        self.load_skin_database()
    
    def load_skin_database(self):
        """Load comprehensive skin database from API"""
        try:
            # Load all skins with detailed information
            skins_url = f"{self.api_base_url}/skins_not_grouped.json"
            response = requests.get(skins_url, timeout=30)
            response.raise_for_status()
            
            skins_data = response.json()
            self.logger.info(f"Loaded {len(skins_data)} skins from database")
            
            # Process each skin
            for skin in skins_data:
                skin_info = self._process_skin_data(skin)
                if skin_info:
                    self.skins_data[skin_info.name] = skin_info
                    
                    # Track weapon types
                    if skin_info.weapon not in self.weapons_list:
                        self.weapons_list.append(skin_info.weapon)
            
            self.logger.info(f"Processed {len(self.skins_data)} skins across {len(self.weapons_list)} weapon types")
            
        except Exception as e:
            self.logger.error(f"Failed to load skin database: {e}")
            self._load_fallback_database()
    
    def _process_skin_data(self, skin_data: Dict) -> Optional[SkinInfo]:
        """Process individual skin data into SkinInfo object"""
        try:
            name = skin_data.get('name', '')
            weapon = skin_data.get('weapon', {}).get('name', '')
            pattern = skin_data.get('pattern', {}).get('name', '')
            rarity = skin_data.get('rarity', {}).get('name', '')
            
            # Get float range
            min_float = float(skin_data.get('min_float', 0.0))
            max_float = float(skin_data.get('max_float', 1.0))
            
            # Calculate wear ranges based on actual float range
            wear_ranges = self._calculate_wear_ranges(min_float, max_float)
            
            # Check for StatTrak and Souvenir variants
            stattrak = skin_data.get('stattrak', False)
            souvenir = skin_data.get('souvenir', False)
            
            return SkinInfo(
                name=name,
                weapon=weapon,
                pattern=pattern,
                rarity=rarity,
                min_float=min_float,
                max_float=max_float,
                wear_ranges=wear_ranges,
                stattrak=stattrak,
                souvenir=souvenir
            )
            
        except Exception as e:
            self.logger.error(f"Error processing skin data: {e}")
            return None
    
    def _calculate_wear_ranges(self, min_float: float, max_float: float) -> Dict[str, Tuple[float, float]]:
        """Calculate actual wear ranges based on skin's float limits"""
        standard_ranges = {
            'Factory New': (0.00, 0.07),
            'Minimal Wear': (0.07, 0.15),
            'Field-Tested': (0.15, 0.37),
            'Well-Worn': (0.37, 0.45),
            'Battle-Scarred': (0.45, 1.00)
        }
        
        actual_ranges = {}
        
        for wear, (std_min, std_max) in standard_ranges.items():
            # Calculate intersection with skin's actual float range
            actual_min = max(min_float, std_min)
            actual_max = min(max_float, std_max)
            
            # Only include wear condition if there's overlap
            if actual_min < actual_max:
                actual_ranges[wear] = (actual_min, actual_max)
        
        return actual_ranges
    
    def _load_fallback_database(self):
        """Load fallback database with known problematic skins"""
        fallback_skins = {
            # AWP Skins with special float ranges
            "AWP | Asiimov": SkinInfo(
                name="AWP | Asiimov",
                weapon="AWP",
                pattern="Asiimov",
                rarity="Covert",
                min_float=0.18,
                max_float=1.00,
                wear_ranges={
                    'Field-Tested': (0.18, 0.37),
                    'Well-Worn': (0.37, 0.45),
                    'Battle-Scarred': (0.45, 1.00)
                }
            ),
            "AWP | Fade": SkinInfo(
                name="AWP | Fade",
                weapon="AWP",
                pattern="Fade",
                rarity="Covert",
                min_float=0.00,
                max_float=0.07,
                wear_ranges={
                    'Factory New': (0.00, 0.07)
                }
            ),
            # M4A4 Skins
            "M4A4 | Howl": SkinInfo(
                name="M4A4 | Howl",
                weapon="M4A4",
                pattern="Howl",
                rarity="Contraband",
                min_float=0.00,
                max_float=1.00,
                wear_ranges={
                    'Factory New': (0.00, 0.04),
                    'Minimal Wear': (0.04, 0.08),
                    'Field-Tested': (0.08, 0.37),
                    'Well-Worn': (0.37, 0.45),
                    'Battle-Scarred': (0.45, 1.00)
                }
            ),
            "M4A4 | Asiimov": SkinInfo(
                name="M4A4 | Asiimov",
                weapon="M4A4",
                pattern="Asiimov",
                rarity="Covert",
                min_float=0.10,
                max_float=1.00,
                wear_ranges={
                    'Field-Tested': (0.15, 0.37),
                    'Well-Worn': (0.37, 0.45),
                    'Battle-Scarred': (0.45, 1.00)
                }
            ),
            # AK-47 Skins
            "AK-47 | Redline": SkinInfo(
                name="AK-47 | Redline",
                weapon="AK-47",
                pattern="Redline",
                rarity="Classified",
                min_float=0.10,
                max_float=1.00,
                wear_ranges={
                    'Field-Tested': (0.15, 0.37),
                    'Well-Worn': (0.37, 0.45),
                    'Battle-Scarred': (0.45, 1.00)
                }
            ),
            # Knife Skins
            "★ Karambit | Rust Coat": SkinInfo(
                name="★ Karambit | Rust Coat",
                weapon="Karambit",
                pattern="Rust Coat",
                rarity="★",
                min_float=0.40,
                max_float=1.00,
                wear_ranges={
                    'Well-Worn': (0.40, 0.45),
                    'Battle-Scarred': (0.45, 1.00)
                }
            )
        }
        
        self.skins_data = fallback_skins
        self.logger.info(f"Loaded {len(fallback_skins)} fallback skins")
    
    def get_skin_info(self, skin_name: str) -> Optional[SkinInfo]:
        """Get detailed information about a specific skin"""
        return self.skins_data.get(skin_name)
    
    def get_all_weapons(self) -> List[str]:
        """Get list of all weapon types"""
        return self.weapons_list.copy()
    
    def get_skins_by_weapon(self, weapon: str) -> List[SkinInfo]:
        """Get all skins for a specific weapon"""
        return [skin for skin in self.skins_data.values() if skin.weapon.lower() == weapon.lower()]
    
    def get_float_capped_skins(self) -> List[SkinInfo]:
        """Get all skins with non-standard float ranges"""
        float_capped = []
        for skin in self.skins_data.values():
            if skin.min_float > 0.0 or skin.max_float < 1.0:
                float_capped.append(skin)
        return float_capped
    
    def get_rare_float_candidates(self, rarity_threshold: float = 0.995) -> List[SkinInfo]:
        """Get skins that can have extremely rare float values"""
        candidates = []
        for skin in self.skins_data.values():
            # Check for very high BS floats
            if skin.max_float >= rarity_threshold:
                candidates.append(skin)
            # Check for very low FN floats
            elif skin.min_float <= 0.005:
                candidates.append(skin)
        return candidates
    
    def is_skin_available_in_wear(self, skin_name: str, wear_condition: str) -> bool:
        """Check if a skin is available in a specific wear condition"""
        skin = self.get_skin_info(skin_name)
        if not skin:
            return False
        return wear_condition in skin.wear_ranges
    
    def get_actual_float_range(self, skin_name: str, wear_condition: str) -> Optional[Tuple[float, float]]:
        """Get the actual float range for a skin in a specific wear condition"""
        skin = self.get_skin_info(skin_name)
        if not skin or wear_condition not in skin.wear_ranges:
            return None
        return skin.wear_ranges[wear_condition]
    
    def search_skins(self, query: str, limit: int = 50) -> List[SkinInfo]:
        """Search for skins by name"""
        query = query.lower()
        results = []
        
        for skin in self.skins_data.values():
            if (query in skin.name.lower() or 
                query in skin.weapon.lower() or 
                query in skin.pattern.lower()):
                results.append(skin)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database"""
        weapon_counts = {}
        rarity_counts = {}
        float_capped_count = 0
        
        for skin in self.skins_data.values():
            # Count by weapon
            weapon_counts[skin.weapon] = weapon_counts.get(skin.weapon, 0) + 1
            
            # Count by rarity
            rarity_counts[skin.rarity] = rarity_counts.get(skin.rarity, 0) + 1
            
            # Count float capped skins
            if skin.min_float > 0.0 or skin.max_float < 1.0:
                float_capped_count += 1
        
        return {
            'total_skins': len(self.skins_data),
            'total_weapons': len(self.weapons_list),
            'float_capped_skins': float_capped_count,
            'weapon_distribution': weapon_counts,
            'rarity_distribution': rarity_counts
        }