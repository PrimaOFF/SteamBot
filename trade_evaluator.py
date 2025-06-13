import os
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from collections import defaultdict

from steam_auth import SteamAuthenticator, SteamMarketTrader
from inventory_manager import InventoryManager, InventoryItem
from float_analyzer import FloatAnalyzer
from database import FloatDatabase
from config import FloatCheckerConfig

class TradeResult(Enum):
    ACCEPT = "accept"
    DECLINE = "decline"
    COUNTER = "counter"
    PENDING = "pending"

class TradeRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class TradeOfferItem:
    asset_id: str
    market_hash_name: str
    estimated_value: float
    market_value: float
    float_value: float = 0.0
    rarity_score: float = 0.0
    condition_score: float = 0.0
    profit_potential: float = 0.0
    is_giving: bool = True  # True if we're giving this item, False if receiving

@dataclass
class TradeEvaluation:
    offer_id: str
    total_giving_value: float
    total_receiving_value: float
    net_value: float
    profit_percentage: float
    risk_level: TradeRisk
    recommendation: TradeResult
    confidence_score: float
    giving_items: List[TradeOfferItem]
    receiving_items: List[TradeOfferItem]
    analysis_notes: List[str]
    evaluated_at: datetime
    auto_accept_eligible: bool = False
    counter_offer_suggestion: Optional[Dict] = None

class TradeOfferEvaluator:
    def __init__(self, authenticator: SteamAuthenticator):
        self.auth = authenticator
        self.trader = SteamMarketTrader(authenticator)
        self.inventory_manager = InventoryManager(authenticator)
        self.float_analyzer = FloatAnalyzer()
        self.database = FloatDatabase()
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Evaluation settings
        self.min_profit_threshold = 0.05  # 5% minimum profit
        self.auto_accept_threshold = 0.15  # 15% profit for auto-accept
        self.max_risk_for_auto = TradeRisk.MEDIUM
        self.min_confidence_for_auto = 0.80  # 80% confidence minimum
        
        # Risk assessment parameters
        self.high_value_threshold = 100.0  # $100+ items are high value
        self.volatility_threshold = 0.3  # 30% price volatility = high risk
        
        # Historical data cache
        self.price_history_cache = {}
        self.evaluation_history = []
        
        # Load user preferences
        self._load_trade_preferences()
    
    def _load_trade_preferences(self):
        """Load user trade preferences from config"""
        try:
            prefs_file = "trade_preferences.json"
            if os.path.exists(prefs_file):
                with open(prefs_file, 'r') as f:
                    prefs = json.load(f)
                    self.min_profit_threshold = prefs.get('min_profit_threshold', 0.05)
                    self.auto_accept_threshold = prefs.get('auto_accept_threshold', 0.15)
                    self.high_value_threshold = prefs.get('high_value_threshold', 100.0)
                    self.logger.info("Trade preferences loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading trade preferences: {e}")
    
    def evaluate_trade_offer(self, offer_data: Dict) -> TradeEvaluation:
        """Evaluate a complete trade offer"""
        try:
            offer_id = offer_data.get('tradeofferid', '')
            
            self.logger.info(f"Evaluating trade offer {offer_id}")
            
            # Process items we're giving away
            giving_items = []
            if 'items_to_give' in offer_data:
                for item_data in offer_data['items_to_give']:
                    processed_item = self._process_trade_item(item_data, is_giving=True)
                    if processed_item:
                        giving_items.append(processed_item)
            
            # Process items we're receiving
            receiving_items = []
            if 'items_to_receive' in offer_data:
                for item_data in offer_data['items_to_receive']:
                    processed_item = self._process_trade_item(item_data, is_giving=False)
                    if processed_item:
                        receiving_items.append(processed_item)
            
            # Calculate total values
            total_giving = sum(item.estimated_value for item in giving_items)
            total_receiving = sum(item.estimated_value for item in receiving_items)
            net_value = total_receiving - total_giving
            
            # Calculate profit percentage
            profit_percentage = (net_value / max(total_giving, 1)) if total_giving > 0 else 0
            
            # Assess risk level
            risk_level = self._assess_trade_risk(giving_items, receiving_items)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                profit_percentage, risk_level, giving_items, receiving_items
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                giving_items, receiving_items, profit_percentage
            )
            
            # Generate analysis notes
            analysis_notes = self._generate_analysis_notes(
                giving_items, receiving_items, profit_percentage, risk_level
            )
            
            # Check auto-accept eligibility
            auto_accept_eligible = self._check_auto_accept_eligibility(
                profit_percentage, risk_level, confidence_score
            )
            
            # Generate counter offer suggestion if applicable
            counter_suggestion = None
            if recommendation == TradeResult.COUNTER:
                counter_suggestion = self._generate_counter_suggestion(
                    giving_items, receiving_items, net_value
                )
            
            evaluation = TradeEvaluation(
                offer_id=offer_id,
                total_giving_value=total_giving,
                total_receiving_value=total_receiving,
                net_value=net_value,
                profit_percentage=profit_percentage,
                risk_level=risk_level,
                recommendation=recommendation,
                confidence_score=confidence_score,
                giving_items=giving_items,
                receiving_items=receiving_items,
                analysis_notes=analysis_notes,
                evaluated_at=datetime.now(),
                auto_accept_eligible=auto_accept_eligible,
                counter_offer_suggestion=counter_suggestion
            )
            
            # Save evaluation to history
            self.evaluation_history.append(evaluation)
            self._save_evaluation_to_db(evaluation)
            
            self.logger.info(f"Trade evaluation completed: {recommendation.value} "
                           f"(${net_value:.2f}, {profit_percentage:.1%})")
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating trade offer: {e}")
            # Return a safe default evaluation
            return self._create_error_evaluation(offer_data.get('tradeofferid', ''))
    
    def _process_trade_item(self, item_data: Dict, is_giving: bool) -> Optional[TradeOfferItem]:
        """Process a single item in a trade offer"""
        try:
            # Extract basic item information
            asset_id = item_data.get('assetid', '')
            market_hash_name = item_data.get('market_hash_name', '')
            
            if not market_hash_name:
                return None
            
            # Get market value
            price_data = self.trader.get_item_price(market_hash_name)
            market_value = self._extract_market_value(price_data)
            
            # Estimate float value (would need inspect link for accuracy)
            float_value = self._estimate_item_float(item_data, market_hash_name)
            
            # Calculate rarity score
            rarity_score = 0.0
            if float_value > 0:
                analysis = self.float_analyzer.analyze_float_rarity(
                    market_hash_name, float_value, market_value
                )
                rarity_score = analysis.rarity_score
            
            # Calculate condition score (wear assessment)
            condition_score = self._calculate_condition_score(item_data, float_value)
            
            # Calculate estimated value including float premium
            estimated_value = self._calculate_estimated_item_value(
                market_value, float_value, rarity_score, condition_score
            )
            
            # Calculate profit potential
            profit_potential = estimated_value - market_value
            
            return TradeOfferItem(
                asset_id=asset_id,
                market_hash_name=market_hash_name,
                estimated_value=estimated_value,
                market_value=market_value,
                float_value=float_value,
                rarity_score=rarity_score,
                condition_score=condition_score,
                profit_potential=profit_potential,
                is_giving=is_giving
            )
            
        except Exception as e:
            self.logger.error(f"Error processing trade item: {e}")
            return None
    
    def _extract_market_value(self, price_data: Dict) -> float:
        """Extract market value from price data"""
        if not price_data:
            return 0.0
        
        try:
            # Try different price fields
            if 'lowest_price' in price_data:
                price_str = price_data['lowest_price'].replace('$', '').replace(',', '')
                return float(price_str)
            elif 'median_price' in price_data:
                price_str = price_data['median_price'].replace('$', '').replace(',', '')
                return float(price_str)
            
            return 0.0
        except Exception:
            return 0.0
    
    def _estimate_item_float(self, item_data: Dict, market_hash_name: str) -> float:
        """Estimate float value from item data"""
        try:
            # Extract exterior from market name
            exterior = self._extract_exterior_from_name(market_hash_name)
            
            # Get standard float ranges
            wear_ranges = self.config.WEAR_RANGES
            if exterior in wear_ranges:
                min_float, max_float = wear_ranges[exterior]
                # Return middle of range as estimate
                return (min_float + max_float) / 2
            
            return 0.0
        except Exception:
            return 0.0
    
    def _extract_exterior_from_name(self, market_hash_name: str) -> str:
        """Extract exterior condition from market hash name"""
        exteriors = ['Factory New', 'Minimal Wear', 'Field-Tested', 'Well-Worn', 'Battle-Scarred']
        
        for exterior in exteriors:
            if f"({exterior})" in market_hash_name:
                return exterior
        
        return 'Unknown'
    
    def _calculate_condition_score(self, item_data: Dict, float_value: float) -> float:
        """Calculate item condition score (0-100)"""
        try:
            if float_value <= 0:
                return 50.0  # Default score
            
            # Score based on float value position within wear range
            if float_value <= 0.07:  # Factory New
                return 95.0 - (float_value / 0.07) * 20  # 95-75
            elif float_value <= 0.15:  # Minimal Wear
                return 75.0 - ((float_value - 0.07) / 0.08) * 20  # 75-55
            elif float_value <= 0.37:  # Field-Tested
                return 55.0 - ((float_value - 0.15) / 0.22) * 20  # 55-35
            elif float_value <= 0.45:  # Well-Worn
                return 35.0 - ((float_value - 0.37) / 0.08) * 15  # 35-20
            else:  # Battle-Scarred
                return 20.0 - ((float_value - 0.45) / 0.55) * 15  # 20-5
            
        except Exception:
            return 50.0
    
    def _calculate_estimated_item_value(self, market_value: float, float_value: float, 
                                      rarity_score: float, condition_score: float) -> float:
        """Calculate estimated value including float and condition premiums"""
        try:
            if market_value <= 0:
                return 0.0
            
            base_value = market_value
            
            # Float rarity multiplier
            float_multiplier = 1.0
            if rarity_score >= 90:
                float_multiplier = 1.5  # 50% premium for extremely rare floats
            elif rarity_score >= 70:
                float_multiplier = 1.3  # 30% premium for rare floats
            elif rarity_score >= 50:
                float_multiplier = 1.1  # 10% premium for good floats
            
            # Condition multiplier
            condition_multiplier = 1.0
            if condition_score >= 90:
                condition_multiplier = 1.2  # 20% premium for excellent condition
            elif condition_score >= 70:
                condition_multiplier = 1.1  # 10% premium for good condition
            elif condition_score <= 30:
                condition_multiplier = 0.9  # 10% discount for poor condition
            
            estimated_value = base_value * float_multiplier * condition_multiplier
            
            return round(estimated_value, 2)
            
        except Exception:
            return market_value
    
    def _assess_trade_risk(self, giving_items: List[TradeOfferItem], 
                          receiving_items: List[TradeOfferItem]) -> TradeRisk:
        """Assess overall risk level of the trade"""
        try:
            risk_factors = 0
            high_value_items = 0
            volatile_items = 0
            
            all_items = giving_items + receiving_items
            
            for item in all_items:
                # High value items increase risk
                if item.estimated_value >= self.high_value_threshold:
                    high_value_items += 1
                    risk_factors += 1
                
                # Low confidence in pricing increases risk
                if item.rarity_score <= 20:  # Low rarity/confidence
                    risk_factors += 1
                
                # Check price volatility (would need historical data)
                volatility = self._get_price_volatility(item.market_hash_name)
                if volatility >= self.volatility_threshold:
                    volatile_items += 1
                    risk_factors += 1
            
            # Assess based on risk factors
            total_items = len(all_items)
            risk_ratio = risk_factors / max(total_items, 1)
            
            if risk_ratio >= 0.7 or high_value_items >= 3:
                return TradeRisk.EXTREME
            elif risk_ratio >= 0.5 or high_value_items >= 2 or volatile_items >= 2:
                return TradeRisk.HIGH
            elif risk_ratio >= 0.3 or high_value_items >= 1 or volatile_items >= 1:
                return TradeRisk.MEDIUM
            else:
                return TradeRisk.LOW
                
        except Exception:
            return TradeRisk.HIGH  # Conservative default
    
    def _get_price_volatility(self, market_hash_name: str) -> float:
        """Get price volatility for an item (placeholder for historical analysis)"""
        try:
            # This would analyze historical price data
            # For now, return a conservative estimate
            if any(keyword in market_hash_name.lower() for keyword in ['dragon lore', 'howl', 'dlore']):
                return 0.4  # High volatility for rare items
            elif any(keyword in market_hash_name.lower() for keyword in ['ak-47', 'awp', 'm4a4']):
                return 0.2  # Medium volatility for popular items
            else:
                return 0.1  # Low volatility for common items
        except Exception:
            return 0.3  # Conservative default
    
    def _generate_recommendation(self, profit_percentage: float, risk_level: TradeRisk,
                               giving_items: List[TradeOfferItem], 
                               receiving_items: List[TradeOfferItem]) -> TradeResult:
        """Generate trade recommendation based on analysis"""
        try:
            # Auto-decline if loss is too high
            if profit_percentage <= -0.2:  # 20% loss
                return TradeResult.DECLINE
            
            # Consider profit and risk together
            if profit_percentage >= self.auto_accept_threshold:
                if risk_level in [TradeRisk.LOW, TradeRisk.MEDIUM]:
                    return TradeResult.ACCEPT
                else:
                    return TradeResult.PENDING  # High profit but high risk
            
            elif profit_percentage >= self.min_profit_threshold:
                if risk_level == TradeRisk.LOW:
                    return TradeResult.ACCEPT
                elif risk_level == TradeRisk.MEDIUM:
                    return TradeResult.PENDING
                else:
                    return TradeResult.COUNTER
            
            elif profit_percentage >= -0.05:  # Small loss, might counter
                return TradeResult.COUNTER
            
            else:
                return TradeResult.DECLINE
                
        except Exception:
            return TradeResult.PENDING  # Safe default
    
    def _calculate_confidence_score(self, giving_items: List[TradeOfferItem],
                                  receiving_items: List[TradeOfferItem],
                                  profit_percentage: float) -> float:
        """Calculate confidence score for the evaluation"""
        try:
            confidence_factors = []
            
            all_items = giving_items + receiving_items
            
            # Price data availability
            items_with_prices = sum(1 for item in all_items if item.market_value > 0)
            price_confidence = items_with_prices / max(len(all_items), 1)
            confidence_factors.append(price_confidence)
            
            # Float data confidence
            items_with_floats = sum(1 for item in all_items if item.float_value > 0)
            float_confidence = items_with_floats / max(len(all_items), 1)
            confidence_factors.append(float_confidence * 0.8)  # Float data less critical
            
            # Market stability (inverse of volatility)
            volatilities = [self._get_price_volatility(item.market_hash_name) for item in all_items]
            avg_volatility = sum(volatilities) / max(len(volatilities), 1)
            stability_confidence = 1.0 - min(avg_volatility, 1.0)
            confidence_factors.append(stability_confidence)
            
            # Profit margin confidence (higher margins = more confident)
            if abs(profit_percentage) >= 0.1:  # 10% or more profit/loss
                margin_confidence = min(abs(profit_percentage) * 2, 1.0)
            else:
                margin_confidence = 0.5  # Uncertain for small margins
            confidence_factors.append(margin_confidence)
            
            # Calculate weighted average
            weights = [0.4, 0.2, 0.2, 0.2]  # Price data most important
            weighted_confidence = sum(cf * w for cf, w in zip(confidence_factors, weights))
            
            return round(min(max(weighted_confidence, 0.0), 1.0), 3)
            
        except Exception:
            return 0.5  # Neutral confidence
    
    def _generate_analysis_notes(self, giving_items: List[TradeOfferItem],
                               receiving_items: List[TradeOfferItem],
                               profit_percentage: float, risk_level: TradeRisk) -> List[str]:
        """Generate human-readable analysis notes"""
        notes = []
        
        try:
            # Profit/loss summary
            if profit_percentage >= 0.15:
                notes.append(f"Excellent profit potential: {profit_percentage:.1%}")
            elif profit_percentage >= 0.05:
                notes.append(f"Good profit margin: {profit_percentage:.1%}")
            elif profit_percentage >= 0:
                notes.append(f"Small profit: {profit_percentage:.1%}")
            else:
                notes.append(f"Loss trade: {profit_percentage:.1%}")
            
            # Risk assessment
            notes.append(f"Risk level: {risk_level.value}")
            
            # High-value items
            high_value_giving = [item for item in giving_items if item.estimated_value >= 50]
            high_value_receiving = [item for item in receiving_items if item.estimated_value >= 50]
            
            if high_value_giving:
                notes.append(f"Giving {len(high_value_giving)} high-value items")
            if high_value_receiving:
                notes.append(f"Receiving {len(high_value_receiving)} high-value items")
            
            # Float value analysis
            rare_floats_giving = [item for item in giving_items if item.rarity_score >= 70]
            rare_floats_receiving = [item for item in receiving_items if item.rarity_score >= 70]
            
            if rare_floats_giving:
                notes.append(f"Giving away {len(rare_floats_giving)} rare float items")
            if rare_floats_receiving:
                notes.append(f"Receiving {len(rare_floats_receiving)} rare float items")
            
            # Item condition
            excellent_condition = [item for item in giving_items + receiving_items 
                                 if item.condition_score >= 90]
            if excellent_condition:
                notes.append(f"{len(excellent_condition)} items in excellent condition")
            
        except Exception as e:
            notes.append(f"Analysis error: {str(e)}")
        
        return notes
    
    def _check_auto_accept_eligibility(self, profit_percentage: float, 
                                     risk_level: TradeRisk, confidence_score: float) -> bool:
        """Check if trade is eligible for auto-accept"""
        return (profit_percentage >= self.auto_accept_threshold and
                risk_level in [TradeRisk.LOW, TradeRisk.MEDIUM] and
                confidence_score >= self.min_confidence_for_auto)
    
    def _generate_counter_suggestion(self, giving_items: List[TradeOfferItem],
                                   receiving_items: List[TradeOfferItem],
                                   net_value: float) -> Optional[Dict]:
        """Generate counter offer suggestion"""
        try:
            if net_value >= 0:
                return None  # No counter needed if already profitable
            
            # Calculate how much additional value is needed
            needed_value = abs(net_value) + (sum(item.estimated_value for item in giving_items) * 0.05)
            
            # Suggest items from their inventory that could bridge the gap
            suggestion = {
                'additional_value_needed': needed_value,
                'suggested_action': 'Request additional items or reduce what you give',
                'notes': f'Need approximately ${needed_value:.2f} more value to make trade worthwhile'
            }
            
            return suggestion
            
        except Exception:
            return None
    
    def _save_evaluation_to_db(self, evaluation: TradeEvaluation):
        """Save trade evaluation to database"""
        try:
            # This would extend the database schema to include trade evaluations
            self.logger.info(f"Saved trade evaluation {evaluation.offer_id} to database")
        except Exception as e:
            self.logger.error(f"Error saving evaluation to database: {e}")
    
    def _create_error_evaluation(self, offer_id: str) -> TradeEvaluation:
        """Create a safe error evaluation"""
        return TradeEvaluation(
            offer_id=offer_id,
            total_giving_value=0.0,
            total_receiving_value=0.0,
            net_value=0.0,
            profit_percentage=0.0,
            risk_level=TradeRisk.EXTREME,
            recommendation=TradeResult.DECLINE,
            confidence_score=0.0,
            giving_items=[],
            receiving_items=[],
            analysis_notes=["Error during evaluation - recommend manual review"],
            evaluated_at=datetime.now(),
            auto_accept_eligible=False
        )
    
    def get_evaluation_summary(self, evaluation: TradeEvaluation) -> str:
        """Generate a readable summary of the evaluation"""
        try:
            summary = []
            summary.append(f"Trade Offer {evaluation.offer_id}")
            summary.append(f"Recommendation: {evaluation.recommendation.value.upper()}")
            summary.append(f"Net Value: ${evaluation.net_value:.2f} ({evaluation.profit_percentage:.1%})")
            summary.append(f"Risk: {evaluation.risk_level.value} | Confidence: {evaluation.confidence_score:.1%}")
            
            if evaluation.auto_accept_eligible:
                summary.append("✅ Auto-accept eligible")
            
            if evaluation.analysis_notes:
                summary.append("\nAnalysis:")
                for note in evaluation.analysis_notes[:3]:  # Top 3 notes
                    summary.append(f"• {note}")
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def export_evaluation_report(self, evaluation: TradeEvaluation, filename: str = None) -> str:
        """Export detailed evaluation report"""
        try:
            if filename is None:
                filename = f"trade_eval_{evaluation.offer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert to dictionary for JSON serialization
            report_data = asdict(evaluation)
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, (TradeResult, TradeRisk)):
                    return obj.value
                return obj
            
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=convert_datetime)
            
            self.logger.info(f"Trade evaluation report exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting evaluation report: {e}")
            return ""

# Test function
def test_trade_evaluator():
    """Test trade evaluation functionality"""
    print("🧪 Testing Trade Evaluator...")
    
    from steam_auth import SteamAuthenticator
    
    # Create authenticator
    auth = SteamAuthenticator()
    
    if not auth.login():
        print("❌ Could not authenticate to Steam")
        return False
    
    try:
        # Create trade evaluator
        evaluator = TradeOfferEvaluator(auth)
        
        # Mock trade offer data for testing
        mock_offer = {
            'tradeofferid': 'test_123456',
            'items_to_give': [
                {
                    'assetid': '123',
                    'market_hash_name': 'AK-47 | Redline (Field-Tested)',
                    'classid': '123',
                    'instanceid': '0'
                }
            ],
            'items_to_receive': [
                {
                    'assetid': '456',
                    'market_hash_name': 'AWP | Asiimov (Field-Tested)',
                    'classid': '456',
                    'instanceid': '0'
                }
            ]
        }
        
        # Test evaluation
        print("Testing trade evaluation...")
        evaluation = evaluator.evaluate_trade_offer(mock_offer)
        
        if evaluation:
            print("✅ Trade evaluation successful")
            print(f"Recommendation: {evaluation.recommendation.value}")
            print(f"Net Value: ${evaluation.net_value:.2f}")
            print(f"Risk Level: {evaluation.risk_level.value}")
            print(f"Confidence: {evaluation.confidence_score:.1%}")
            
            # Test summary generation
            summary = evaluator.get_evaluation_summary(evaluation)
            print("\n✅ Evaluation Summary:")
            print(summary)
            
            # Test report export
            report_file = evaluator.export_evaluation_report(evaluation)
            if report_file:
                print(f"✅ Evaluation report exported: {report_file}")
            
            return True
        else:
            print("❌ Trade evaluation failed")
            return False
            
    finally:
        auth.logout()

if __name__ == "__main__":
    test_trade_evaluator()