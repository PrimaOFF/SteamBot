import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from config import FloatCheckerConfig
from skin_database import SkinDatabase, SkinInfo

@dataclass
class FloatAnalysis:
    item_name: str
    wear_condition: str
    float_value: float
    price: float
    rarity_score: float
    is_rare: bool
    analysis_timestamp: datetime
    inspect_link: str = ""
    market_url: str = ""

class FloatAnalyzer:
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        self.skin_db = SkinDatabase()
    
    def analyze_float_rarity(self, item_name: str, float_value: float, price: float) -> FloatAnalysis:
        """Analyze the rarity of a float value for a specific item"""
        wear_condition = self.determine_wear_condition(item_name, float_value)
        rarity_score = self.calculate_rarity_score(item_name, float_value, wear_condition)
        is_rare = self.is_float_rare(item_name, float_value, wear_condition)
        
        return FloatAnalysis(
            item_name=item_name,
            wear_condition=wear_condition,
            float_value=float_value,
            price=price,
            rarity_score=rarity_score,
            is_rare=is_rare,
            analysis_timestamp=datetime.now()
        )
    
    def determine_wear_condition(self, item_name: str, float_value: float) -> str:
        """Determine wear condition based on float value"""
        # Get skin info from database
        skin_info = self.skin_db.get_skin_info(item_name)
        
        if skin_info and skin_info.wear_ranges:
            # Use actual skin-specific ranges
            for wear, (min_val, max_val) in skin_info.wear_ranges.items():
                if min_val <= float_value < max_val:
                    return wear
        else:
            # Fallback to config ranges
            if item_name in self.config.SKIN_SPECIFIC_RANGES:
                ranges = self.config.SKIN_SPECIFIC_RANGES[item_name]
            else:
                ranges = self.config.WEAR_RANGES
            
            for wear, (min_val, max_val) in ranges.items():
                if min_val <= float_value < max_val:
                    return wear
        
        return "Unknown"
    
    def calculate_rarity_score(self, item_name: str, float_value: float, wear_condition: str) -> float:
        """Calculate a rarity score (0-100) based on float value"""
        skin_info = self.skin_db.get_skin_info(item_name)
        
        if wear_condition == "Factory New":
            # Get actual FN range for this skin
            if skin_info and 'Factory New' in skin_info.wear_ranges:
                min_fn, max_fn = skin_info.wear_ranges['Factory New']
                
                # Calculate percentage within FN range
                fn_position = (float_value - min_fn) / (max_fn - min_fn) if max_fn > min_fn else 0
                
                # Lower float = higher rarity for FN
                if fn_position <= 0.01:  # Bottom 1%
                    return 100.0
                elif fn_position <= 0.05:  # Bottom 5%
                    return 95.0
                elif fn_position <= 0.10:  # Bottom 10%
                    return 85.0
                elif fn_position <= 0.20:  # Bottom 20%
                    return 70.0
                else:
                    return 40.0
            else:
                # Fallback to standard thresholds
                if float_value <= 0.001:
                    return 100.0
                elif float_value <= 0.005:
                    return 90.0
                elif float_value <= 0.01:
                    return 75.0
                else:
                    return 40.0
        
        elif wear_condition == "Battle-Scarred":
            # Get actual BS range for this skin
            if skin_info and 'Battle-Scarred' in skin_info.wear_ranges:
                min_bs, max_bs = skin_info.wear_ranges['Battle-Scarred']
                
                # Calculate percentage within BS range
                bs_position = (float_value - min_bs) / (max_bs - min_bs) if max_bs > min_bs else 0
                
                # Higher float = higher rarity for BS
                if bs_position >= 0.99:  # Top 1%
                    return 100.0
                elif bs_position >= 0.95:  # Top 5%
                    return 95.0
                elif bs_position >= 0.90:  # Top 10%
                    return 85.0
                elif bs_position >= 0.80:  # Top 20%
                    return 70.0
                else:
                    return 40.0
            else:
                # Fallback to standard thresholds
                if float_value >= 0.999:
                    return 100.0
                elif float_value >= 0.995:
                    return 90.0
                elif float_value >= 0.99:
                    return 75.0
                else:
                    return 40.0
        
        else:
            # For other wear conditions, calculate based on position within range
            if skin_info and wear_condition in skin_info.wear_ranges:
                min_val, max_val = skin_info.wear_ranges[wear_condition]
            elif item_name in self.config.SKIN_SPECIFIC_RANGES and wear_condition in self.config.SKIN_SPECIFIC_RANGES[item_name]:
                min_val, max_val = self.config.SKIN_SPECIFIC_RANGES[item_name][wear_condition]
            elif wear_condition in self.config.WEAR_RANGES:
                min_val, max_val = self.config.WEAR_RANGES[wear_condition]
            else:
                return 0.0
            
            position = (float_value - min_val) / (max_val - min_val) if max_val > min_val else 0
            
            if position <= 0.05 or position >= 0.95:
                return 75.0  # Extreme edges
            elif position <= 0.15 or position >= 0.85:
                return 60.0  # Near edges
            elif position <= 0.25 or position >= 0.75:
                return 45.0  # Somewhat rare
            else:
                return 25.0  # Common
        
        return 0.0  # Unknown
    
    def is_float_rare(self, item_name: str, float_value: float, wear_condition: str) -> bool:
        """Determine if a float value is considered rare"""
        rarity_score = self.calculate_rarity_score(item_name, float_value, wear_condition)
        return rarity_score >= 70.0
    
    def get_float_percentile(self, item_name: str, float_value: float, wear_condition: str) -> float:
        """Get the percentile of this float value within its wear condition"""
        # This would require historical data analysis
        # For now, return a calculated estimate
        if wear_condition == "Factory New":
            if float_value <= 0.001:
                return 99.9
            elif float_value <= 0.005:
                return 99.0
            elif float_value <= 0.01:
                return 95.0
            elif float_value <= 0.02:
                return 85.0
            else:
                return 50.0
        
        elif wear_condition == "Battle-Scarred":
            if float_value >= 0.999:
                return 99.9
            elif float_value >= 0.995:
                return 99.0
            elif float_value >= 0.99:
                return 95.0
            elif float_value >= 0.98:
                return 85.0
            else:
                return 50.0
        
        return 50.0  # Default middle percentile
    
    def compare_to_market_average(self, item_name: str, float_value: float, price: float) -> Dict:
        """Compare item to market average for similar float values"""
        # This would require market data analysis
        # For now, return placeholder data
        return {
            'price_compared_to_average': 0.0,
            'float_compared_to_average': 0.0,
            'value_rating': 'Unknown'
        }
    
    def generate_investment_recommendation(self, analysis: FloatAnalysis) -> str:
        """Generate investment recommendation based on analysis"""
        if analysis.rarity_score >= 90:
            return "STRONG BUY - Extremely rare float value"
        elif analysis.rarity_score >= 70:
            return "BUY - Rare float value with good potential"
        elif analysis.rarity_score >= 50:
            return "HOLD - Moderate rarity, watch for price changes"
        elif analysis.rarity_score >= 30:
            return "NEUTRAL - Common float, standard market value"
        else:
            return "AVOID - Very common float, poor investment potential"
    
    def batch_analyze_items(self, items_data: List[Dict]) -> List[FloatAnalysis]:
        """Analyze multiple items at once"""
        analyses = []
        
        for item_data in items_data:
            if all(key in item_data for key in ['name', 'float', 'price']):
                analysis = self.analyze_float_rarity(
                    item_data['name'],
                    item_data['float'],
                    item_data['price']
                )
                analyses.append(analysis)
        
        return analyses
    
    def filter_rare_items(self, analyses: List[FloatAnalysis], min_rarity_score: float = 70.0) -> List[FloatAnalysis]:
        """Filter analyses to only include rare items"""
        return [analysis for analysis in analyses if analysis.rarity_score >= min_rarity_score]