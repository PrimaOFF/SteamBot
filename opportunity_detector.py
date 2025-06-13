import os
import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict
import pandas as pd
import statistics

from steam_auth import SteamAuthenticator, SteamMarketTrader
from inventory_manager import InventoryManager, InventoryItem
from price_tracker import PriceTracker, MarketTrend, PriceChangeType
from trade_evaluator import TradeOfferEvaluator, TradeResult, TradeRisk
from float_analyzer import FloatAnalyzer
from optimized_steam_api import OptimizedSteamAPI
from database import FloatDatabase
from config import FloatCheckerConfig

class OpportunityType(Enum):
    ARBITRAGE = "arbitrage"
    FLOAT_UNDERVALUED = "float_undervalued"
    PRICE_DROP = "price_drop"
    VOLUME_ANOMALY = "volume_anomaly"
    PATTERN_BREAKOUT = "pattern_breakout"
    CROSS_PLATFORM = "cross_platform"
    INVENTORY_OPTIMIZATION = "inventory_optimization"

class OpportunityPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class MarketOpportunity:
    opportunity_id: str
    opportunity_type: OpportunityType
    item_name: str
    current_price: float
    target_price: float
    profit_potential: float
    profit_percentage: float
    priority: OpportunityPriority
    confidence_score: float
    risk_level: TradeRisk
    time_sensitivity: int  # Hours until opportunity expires
    action_required: str
    analysis_data: Dict
    detected_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

@dataclass
class ArbitrageOpportunity:
    item_name: str
    buy_platform: str
    sell_platform: str
    buy_price: float
    sell_price: float
    profit: float
    profit_percentage: float
    volume_available: int
    execution_complexity: int  # 1-10 scale
    estimated_time: int  # Minutes to execute
    
@dataclass
class PortfolioOptimization:
    total_portfolio_value: float
    suggested_sells: List[InventoryItem]
    suggested_buys: List[str]
    expected_profit: float
    optimization_score: float
    rebalancing_strategy: str

class OpportunityDetector:
    def __init__(self, authenticator: SteamAuthenticator):
        self.auth = authenticator
        self.trader = SteamMarketTrader(authenticator)
        self.inventory_manager = InventoryManager(authenticator)
        self.price_tracker = PriceTracker(authenticator)
        self.trade_evaluator = TradeOfferEvaluator(authenticator)
        self.float_analyzer = FloatAnalyzer()
        self.api = OptimizedSteamAPI()
        self.database = FloatDatabase()
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Detection settings
        self.min_profit_threshold = 5.0  # $5 minimum profit
        self.min_profit_percentage = 0.10  # 10% minimum profit margin
        self.max_risk_tolerance = TradeRisk.MEDIUM
        self.opportunity_scan_interval = 1800  # 30 minutes
        
        # Opportunity storage
        self.active_opportunities = {}
        self.opportunity_history = []
        
        # Detection algorithms
        self.detection_algorithms = {
            OpportunityType.ARBITRAGE: self._detect_arbitrage_opportunities,
            OpportunityType.FLOAT_UNDERVALUED: self._detect_float_undervalued,
            OpportunityType.PRICE_DROP: self._detect_price_drops,
            OpportunityType.VOLUME_ANOMALY: self._detect_volume_anomalies,
            OpportunityType.PATTERN_BREAKOUT: self._detect_pattern_breakouts,
            OpportunityType.INVENTORY_OPTIMIZATION: self._detect_inventory_optimization
        }
        
        # Scanning state
        self.is_scanning = False
        self.scanning_thread = None
        
        # Load configuration
        self._load_detector_config()
    
    def _load_detector_config(self):
        """Load opportunity detector configuration"""
        try:
            config_file = "opportunity_detector_config.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.min_profit_threshold = config.get('min_profit_threshold', 5.0)
                    self.min_profit_percentage = config.get('min_profit_percentage', 0.10)
                    self.opportunity_scan_interval = config.get('scan_interval', 1800)
                    self.logger.info("Opportunity detector configuration loaded")
        except Exception as e:
            self.logger.error(f"Error loading detector config: {e}")
    
    def start_scanning(self):
        """Start continuous opportunity scanning"""
        if self.is_scanning:
            self.logger.warning("Opportunity scanning already running")
            return
        
        try:
            self.is_scanning = True
            self.scanning_thread = threading.Thread(target=self._scanning_loop, daemon=True)
            self.scanning_thread.start()
            self.logger.info("Opportunity scanning started")
            
        except Exception as e:
            self.logger.error(f"Error starting opportunity scanning: {e}")
            self.is_scanning = False
    
    def stop_scanning(self):
        """Stop opportunity scanning"""
        self.is_scanning = False
        if self.scanning_thread:
            self.scanning_thread.join(timeout=15)
        self.logger.info("Opportunity scanning stopped")
    
    def _scanning_loop(self):
        """Main opportunity scanning loop"""
        while self.is_scanning:
            try:
                # Run all detection algorithms
                self.scan_for_opportunities()
                
                # Clean up expired opportunities
                self._cleanup_expired_opportunities()
                
                # Wait for next scan
                time.sleep(self.opportunity_scan_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scanning loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def scan_for_opportunities(self) -> List[MarketOpportunity]:
        """Scan for all types of market opportunities"""
        try:
            self.logger.info("Starting comprehensive opportunity scan")
            
            all_opportunities = []
            
            # Run each detection algorithm
            for opp_type, algorithm in self.detection_algorithms.items():
                try:
                    opportunities = algorithm()
                    if opportunities:
                        all_opportunities.extend(opportunities)
                        self.logger.info(f"Found {len(opportunities)} {opp_type.value} opportunities")
                except Exception as e:
                    self.logger.error(f"Error in {opp_type.value} detection: {e}")
            
            # Filter and prioritize opportunities
            filtered_opportunities = self._filter_opportunities(all_opportunities)
            
            # Update active opportunities
            for opportunity in filtered_opportunities:
                self.active_opportunities[opportunity.opportunity_id] = opportunity
            
            self.logger.info(f"Scan completed: {len(filtered_opportunities)} opportunities detected")
            
            return filtered_opportunities
            
        except Exception as e:
            self.logger.error(f"Error scanning for opportunities: {e}")
            return []
    
    def _detect_arbitrage_opportunities(self) -> List[MarketOpportunity]:
        """Detect arbitrage opportunities between different platforms/markets"""
        opportunities = []
        
        try:
            # This would compare prices across different platforms
            # For now, simulate some basic arbitrage detection
            
            high_volume_items = [
                "AK-47 | Redline (Field-Tested)",
                "AWP | Asiimov (Field-Tested)",
                "M4A4 | Desolate Space (Field-Tested)",
                "Glock-18 | Water Elemental (Minimal Wear)"
            ]
            
            for item_name in high_volume_items:
                # Get current Steam Market price
                steam_price = self.price_tracker.get_current_price(item_name)
                
                if steam_price > 0:
                    # Simulate third-party platform price (would be real API call)
                    third_party_price = steam_price * (0.85 + (hash(item_name) % 20) / 100)  # 85-105% of Steam price
                    
                    if steam_price > third_party_price * 1.15:  # 15% or more difference
                        profit = steam_price - third_party_price - (steam_price * 0.13)  # Steam 13% fee
                        profit_percentage = profit / third_party_price
                        
                        if profit >= self.min_profit_threshold and profit_percentage >= self.min_profit_percentage:
                            opportunity = MarketOpportunity(
                                opportunity_id=f"arbitrage_{item_name}_{int(time.time())}",
                                opportunity_type=OpportunityType.ARBITRAGE,
                                item_name=item_name,
                                current_price=third_party_price,
                                target_price=steam_price * 0.87,  # After Steam fees
                                profit_potential=profit,
                                profit_percentage=profit_percentage,
                                priority=self._calculate_priority(profit, profit_percentage),
                                confidence_score=0.7,  # Moderate confidence for arbitrage
                                risk_level=TradeRisk.MEDIUM,
                                time_sensitivity=6,  # 6 hours before prices might change
                                action_required=f"Buy on third-party platform at ${third_party_price:.2f}, sell on Steam",
                                analysis_data={
                                    'steam_price': steam_price,
                                    'third_party_price': third_party_price,
                                    'estimated_fees': steam_price * 0.13
                                },
                                detected_at=datetime.now(),
                                expires_at=datetime.now() + timedelta(hours=6)
                            )
                            opportunities.append(opportunity)
            
        except Exception as e:
            self.logger.error(f"Error detecting arbitrage opportunities: {e}")
        
        return opportunities
    
    def _detect_float_undervalued(self) -> List[MarketOpportunity]:
        """Detect items with rare floats that are undervalued"""
        opportunities = []
        
        try:
            # Get recent float analysis results from database
            recent_analyses = self.database.get_recent_rare_floats(limit=50)
            
            for analysis in recent_analyses:
                if analysis.rarity_score >= 70:  # High rarity float
                    current_price = self.price_tracker.get_current_price(analysis.item_name)
                    
                    if current_price > 0:
                        # Calculate expected price based on float rarity
                        expected_premium = 1.0
                        if analysis.rarity_score >= 95:
                            expected_premium = 2.0  # 100% premium for extremely rare
                        elif analysis.rarity_score >= 85:
                            expected_premium = 1.5  # 50% premium for very rare
                        elif analysis.rarity_score >= 70:
                            expected_premium = 1.2  # 20% premium for rare
                        
                        expected_price = current_price * expected_premium
                        profit_potential = expected_price - current_price
                        profit_percentage = (expected_price - current_price) / current_price
                        
                        if profit_potential >= self.min_profit_threshold and profit_percentage >= 0.15:
                            opportunity = MarketOpportunity(
                                opportunity_id=f"float_undervalued_{analysis.item_name}_{analysis.float_value}_{int(time.time())}",
                                opportunity_type=OpportunityType.FLOAT_UNDERVALUED,
                                item_name=analysis.item_name,
                                current_price=current_price,
                                target_price=expected_price,
                                profit_potential=profit_potential,
                                profit_percentage=profit_percentage,
                                priority=self._calculate_priority(profit_potential, profit_percentage),
                                confidence_score=analysis.rarity_score / 100,
                                risk_level=TradeRisk.LOW if analysis.rarity_score >= 90 else TradeRisk.MEDIUM,
                                time_sensitivity=24,  # 24 hours to act
                                action_required=f"Purchase rare float item (float: {analysis.float_value:.6f})",
                                analysis_data={
                                    'float_value': analysis.float_value,
                                    'rarity_score': analysis.rarity_score,
                                    'expected_premium': expected_premium,
                                    'market_hash_name': analysis.item_name
                                },
                                detected_at=datetime.now(),
                                expires_at=datetime.now() + timedelta(hours=24)
                            )
                            opportunities.append(opportunity)
            
        except Exception as e:
            self.logger.error(f"Error detecting float undervalued opportunities: {e}")
        
        return opportunities
    
    def _detect_price_drops(self) -> List[MarketOpportunity]:
        """Detect significant price drops that represent buying opportunities"""
        opportunities = []
        
        try:
            # Check price trends for all tracked items
            for item_name in self.price_tracker.watched_items:
                trend = self.price_tracker.calculate_market_trend(item_name)
                
                if trend and trend.change_24h <= -0.15:  # 15% or more drop
                    current_price = trend.current_price
                    expected_recovery_price = trend.price_24h_ago * 0.95  # Expect 95% recovery
                    
                    profit_potential = expected_recovery_price - current_price
                    profit_percentage = profit_potential / current_price
                    
                    if profit_potential >= self.min_profit_threshold:
                        # Assess if this is a temporary dip vs permanent decline
                        confidence = self._assess_recovery_confidence(trend)
                        
                        if confidence >= 0.6:  # 60% confidence in recovery
                            opportunity = MarketOpportunity(
                                opportunity_id=f"price_drop_{item_name}_{int(time.time())}",
                                opportunity_type=OpportunityType.PRICE_DROP,
                                item_name=item_name,
                                current_price=current_price,
                                target_price=expected_recovery_price,
                                profit_potential=profit_potential,
                                profit_percentage=profit_percentage,
                                priority=self._calculate_priority(profit_potential, profit_percentage),
                                confidence_score=confidence,
                                risk_level=TradeRisk.MEDIUM if confidence >= 0.8 else TradeRisk.HIGH,
                                time_sensitivity=48,  # 48 hours for recovery
                                action_required=f"Buy during price dip (down {abs(trend.change_24h):.1%})",
                                analysis_data={
                                    'price_drop_24h': trend.change_24h,
                                    'price_drop_7d': trend.change_7d,
                                    'volatility_score': trend.volatility_score,
                                    'recovery_confidence': confidence
                                },
                                detected_at=datetime.now(),
                                expires_at=datetime.now() + timedelta(hours=48)
                            )
                            opportunities.append(opportunity)
            
        except Exception as e:
            self.logger.error(f"Error detecting price drop opportunities: {e}")
        
        return opportunities
    
    def _detect_volume_anomalies(self) -> List[MarketOpportunity]:
        """Detect unusual volume patterns that might indicate opportunities"""
        opportunities = []
        
        try:
            # This would analyze volume spikes and drops
            # For now, return empty list as volume data isn't fully implemented
            pass
            
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {e}")
        
        return opportunities
    
    def _detect_pattern_breakouts(self) -> List[MarketOpportunity]:
        """Detect technical pattern breakouts"""
        opportunities = []
        
        try:
            # This would implement technical analysis patterns
            # For now, return empty list as this requires extensive price history
            pass
            
        except Exception as e:
            self.logger.error(f"Error detecting pattern breakouts: {e}")
        
        return opportunities
    
    def _detect_inventory_optimization(self) -> List[MarketOpportunity]:
        """Detect opportunities to optimize current inventory"""
        opportunities = []
        
        try:
            # Get portfolio summary
            portfolio = self.inventory_manager.get_portfolio_summary()
            
            if portfolio.total_items == 0:
                return opportunities
            
            # Look for items with high profit potential
            for item in portfolio.trading_candidates[:5]:  # Top 5 candidates
                if item.profit_potential >= self.min_profit_threshold:
                    opportunity = MarketOpportunity(
                        opportunity_id=f"inventory_opt_{item.asset_id}_{int(time.time())}",
                        opportunity_type=OpportunityType.INVENTORY_OPTIMIZATION,
                        item_name=item.market_hash_name,
                        current_price=item.market_price,
                        target_price=item.estimated_price,
                        profit_potential=item.profit_potential,
                        profit_percentage=item.profit_potential / max(item.market_price, 1),
                        priority=self._calculate_priority(item.profit_potential, item.profit_potential / max(item.market_price, 1)),
                        confidence_score=0.8,  # High confidence for owned items
                        risk_level=TradeRisk.LOW,  # Low risk for items we own
                        time_sensitivity=168,  # 1 week to act
                        action_required=f"Sell inventory item for profit",
                        analysis_data={
                            'asset_id': item.asset_id,
                            'estimated_value': item.estimated_price,
                            'market_value': item.market_price,
                            'float_value': item.float_value,
                            'rarity_score': getattr(item, 'rarity_score', 0)
                        },
                        detected_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(days=7)
                    )
                    opportunities.append(opportunity)
            
        except Exception as e:
            self.logger.error(f"Error detecting inventory optimization opportunities: {e}")
        
        return opportunities
    
    def _calculate_priority(self, profit_potential: float, profit_percentage: float) -> OpportunityPriority:
        """Calculate opportunity priority based on profit metrics"""
        try:
            if profit_potential >= 50 and profit_percentage >= 0.5:  # $50+ and 50%+
                return OpportunityPriority.CRITICAL
            elif profit_potential >= 20 and profit_percentage >= 0.25:  # $20+ and 25%+
                return OpportunityPriority.HIGH
            elif profit_potential >= 10 and profit_percentage >= 0.15:  # $10+ and 15%+
                return OpportunityPriority.MEDIUM
            else:
                return OpportunityPriority.LOW
        except Exception:
            return OpportunityPriority.LOW
    
    def _assess_recovery_confidence(self, trend: MarketTrend) -> float:
        """Assess confidence that a price will recover from a drop"""
        try:
            confidence_factors = []
            
            # Volatility factor (lower volatility = higher confidence)
            volatility_factor = max(0, 1 - trend.volatility_score)
            confidence_factors.append(volatility_factor * 0.3)
            
            # Magnitude factor (smaller drops more likely to recover)
            drop_magnitude = abs(trend.change_24h)
            magnitude_factor = max(0, 1 - (drop_magnitude - 0.15) / 0.35)  # Scale from 15% to 50% drop
            confidence_factors.append(magnitude_factor * 0.3)
            
            # Trend consistency (gradual vs sudden drop)
            if trend.change_7d != 0:
                consistency_factor = min(abs(trend.change_24h / trend.change_7d), 1.0)
            else:
                consistency_factor = 0.5
            confidence_factors.append((1 - consistency_factor) * 0.2)  # Sudden drops more likely to recover
            
            # Data quality factor
            data_quality = trend.confidence_score
            confidence_factors.append(data_quality * 0.2)
            
            return min(sum(confidence_factors), 1.0)
            
        except Exception:
            return 0.5  # Neutral confidence
    
    def _filter_opportunities(self, opportunities: List[MarketOpportunity]) -> List[MarketOpportunity]:
        """Filter and deduplicate opportunities"""
        try:
            # Remove duplicates and low-value opportunities
            filtered = []
            seen_items = set()
            
            # Sort by priority and profit potential
            sorted_opportunities = sorted(
                opportunities,
                key=lambda x: (x.priority.value, -x.profit_potential),
                reverse=False
            )
            
            for opp in sorted_opportunities:
                # Skip if we already have an opportunity for this item
                if opp.item_name in seen_items:
                    continue
                
                # Skip if below minimum thresholds
                if opp.profit_potential < self.min_profit_threshold:
                    continue
                
                if opp.profit_percentage < self.min_profit_percentage:
                    continue
                
                # Skip if risk is too high
                if opp.risk_level == TradeRisk.EXTREME:
                    continue
                
                filtered.append(opp)
                seen_items.add(opp.item_name)
            
            return filtered[:20]  # Return top 20 opportunities
            
        except Exception as e:
            self.logger.error(f"Error filtering opportunities: {e}")
            return opportunities
    
    def _cleanup_expired_opportunities(self):
        """Remove expired opportunities"""
        try:
            current_time = datetime.now()
            expired_ids = []
            
            for opp_id, opportunity in self.active_opportunities.items():
                if opportunity.expires_at and current_time > opportunity.expires_at:
                    expired_ids.append(opp_id)
                    opportunity.is_active = False
            
            for opp_id in expired_ids:
                del self.active_opportunities[opp_id]
            
            if expired_ids:
                self.logger.info(f"Cleaned up {len(expired_ids)} expired opportunities")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up opportunities: {e}")
    
    def get_active_opportunities(self, opportunity_type: OpportunityType = None, 
                               priority: OpportunityPriority = None) -> List[MarketOpportunity]:
        """Get active opportunities with optional filtering"""
        try:
            opportunities = list(self.active_opportunities.values())
            
            # Filter by type
            if opportunity_type:
                opportunities = [opp for opp in opportunities if opp.opportunity_type == opportunity_type]
            
            # Filter by priority
            if priority:
                opportunities = [opp for opp in opportunities if opp.priority == priority]
            
            # Sort by priority and profit potential
            opportunities.sort(key=lambda x: (x.priority.value, -x.profit_potential))
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error getting active opportunities: {e}")
            return []
    
    def get_opportunity_summary(self) -> Dict:
        """Get summary of current opportunities"""
        try:
            opportunities = list(self.active_opportunities.values())
            
            summary = {
                'total_opportunities': len(opportunities),
                'by_type': {},
                'by_priority': {},
                'total_profit_potential': sum(opp.profit_potential for opp in opportunities),
                'avg_confidence': statistics.mean([opp.confidence_score for opp in opportunities]) if opportunities else 0,
                'high_priority_count': len([opp for opp in opportunities if opp.priority in [OpportunityPriority.CRITICAL, OpportunityPriority.HIGH]]),
                'is_scanning': self.is_scanning
            }
            
            # Count by type
            for opp_type in OpportunityType:
                count = len([opp for opp in opportunities if opp.opportunity_type == opp_type])
                summary['by_type'][opp_type.value] = count
            
            # Count by priority
            for priority in OpportunityPriority:
                count = len([opp for opp in opportunities if opp.priority == priority])
                summary['by_priority'][priority.value] = count
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting opportunity summary: {e}")
            return {}
    
    def export_opportunities(self, filename: str = None) -> str:
        """Export opportunities to JSON file"""
        try:
            if filename is None:
                filename = f"market_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            opportunities_data = []
            for opportunity in self.active_opportunities.values():
                opp_dict = asdict(opportunity)
                # Convert datetime objects to strings
                for key, value in opp_dict.items():
                    if isinstance(value, datetime):
                        opp_dict[key] = value.isoformat()
                    elif hasattr(value, 'value'):  # Enum
                        opp_dict[key] = value.value
                opportunities_data.append(opp_dict)
            
            export_data = {
                'opportunities': opportunities_data,
                'summary': self.get_opportunity_summary(),
                'generated_at': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Opportunities exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting opportunities: {e}")
            return ""

# Test function
def test_opportunity_detector():
    """Test opportunity detection functionality"""
    print("🧪 Testing Opportunity Detector...")
    
    from steam_auth import SteamAuthenticator
    
    # Create authenticator
    auth = SteamAuthenticator()
    
    if not auth.login():
        print("❌ Could not authenticate to Steam")
        return False
    
    try:
        # Create opportunity detector
        detector = OpportunityDetector(auth)
        
        # Test opportunity scanning
        print("Scanning for opportunities...")
        opportunities = detector.scan_for_opportunities()
        
        if opportunities:
            print(f"✅ Found {len(opportunities)} opportunities")
            
            # Show top opportunities
            for i, opp in enumerate(opportunities[:3]):
                print(f"   {i+1}. {opp.opportunity_type.value}: {opp.item_name}")
                print(f"      Profit: ${opp.profit_potential:.2f} ({opp.profit_percentage:.1%})")
                print(f"      Priority: {opp.priority.value}")
        else:
            print("No opportunities found (this is normal for test data)")
        
        # Test summary
        summary = detector.get_opportunity_summary()
        print("✅ Opportunity Summary:")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        # Test export
        export_file = detector.export_opportunities()
        if export_file:
            print(f"✅ Opportunities exported to {export_file}")
        
        print("✅ Opportunity detector test completed")
        return True
        
    finally:
        auth.logout()

if __name__ == "__main__":
    test_opportunity_detector()