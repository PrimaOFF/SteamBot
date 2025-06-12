import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from config import FloatCheckerConfig

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
        # Check if item has specific ranges
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
        if wear_condition == "Factory New":
            # Lower float = higher rarity for FN
            if float_value <= self.config.RARE_FLOAT_THRESHOLDS['factory_new']['extreme_max']:
                return 100.0  # Extremely rare
            elif float_value <= self.config.RARE_FLOAT_THRESHOLDS['factory_new']['max']:
                return 90.0   # Very rare
            elif float_value <= 0.01:
                return 75.0   # Rare
            elif float_value <= 0.02:
                return 60.0   # Uncommon
            else:
                return 30.0   # Common FN
        
        elif wear_condition == "Battle-Scarred":
            # Higher float = higher rarity for BS
            if float_value >= self.config.RARE_FLOAT_THRESHOLDS['battle_scarred']['extreme_min']:
                return 100.0  # Extremely rare
            elif float_value >= self.config.RARE_FLOAT_THRESHOLDS['battle_scarred']['min']:
                return 90.0   # Very rare
            elif float_value >= 0.99:
                return 75.0   # Rare
            elif float_value >= 0.98:
                return 60.0   # Uncommon
            else:
                return 30.0   # Common BS
        
        else:
            # For other wear conditions, calculate based on position within range
            if item_name in self.config.SKIN_SPECIFIC_RANGES:
                ranges = self.config.SKIN_SPECIFIC_RANGES[item_name]
            else:
                ranges = self.config.WEAR_RANGES
            
            if wear_condition in ranges:
                min_val, max_val = ranges[wear_condition]
                position = (float_value - min_val) / (max_val - min_val)
                
                if position <= 0.1 or position >= 0.9:
                    return 70.0  # Edge cases within range
                elif position <= 0.2 or position >= 0.8:
                    return 50.0  # Near edges
                else:
                    return 25.0  # Middle of range
        
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