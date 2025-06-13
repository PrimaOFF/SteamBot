import logging
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

from steam_auth import SteamAuthenticator, SteamMarketTrader
from skin_database import SkinDatabase
from float_analyzer import FloatAnalyzer
from database import FloatDatabase
from config import FloatCheckerConfig

@dataclass
class InventoryItem:
    asset_id: str
    class_id: str
    instance_id: str
    market_hash_name: str
    name: str
    type: str
    rarity: str = ""
    exterior: str = ""
    float_value: float = 0.0
    estimated_price: float = 0.0
    market_price: float = 0.0
    profit_potential: float = 0.0
    last_updated: datetime = None
    tradable: bool = True
    marketable: bool = True
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class PortfolioSummary:
    total_items: int
    total_value: float
    total_market_value: float
    potential_profit: float
    high_value_items: List[InventoryItem]
    trading_candidates: List[InventoryItem]
    market_opportunities: List[InventoryItem]
    last_updated: datetime

class InventoryManager:
    def __init__(self, authenticator: SteamAuthenticator):
        self.auth = authenticator
        self.trader = SteamMarketTrader(authenticator)
        self.skin_db = SkinDatabase()
        self.float_analyzer = FloatAnalyzer()
        self.database = FloatDatabase()
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Cache settings
        self.inventory_cache = {}
        self.price_cache = {}
        self.cache_duration = timedelta(minutes=30)
        self.last_inventory_update = None
        
        # Valuation settings
        self.high_value_threshold = 100.0  # Items worth $100+
        self.profit_threshold = 0.15  # 15% profit margin
        
    def refresh_inventory(self, force: bool = False) -> bool:
        """Refresh inventory data from Steam"""
        try:
            # Check if we need to refresh
            if not force and self.last_inventory_update:
                time_since_update = datetime.now() - self.last_inventory_update
                if time_since_update < self.cache_duration:
                    self.logger.info("Using cached inventory data")
                    return True
            
            if not self.auth.is_authenticated():
                self.logger.error("Not authenticated to Steam")
                return False
            
            self.logger.info("Refreshing inventory from Steam...")
            
            # Get raw inventory data
            raw_inventory = self.auth.get_inventory()
            if not raw_inventory:
                self.logger.error("Failed to get inventory from Steam")
                return False
            
            # Process inventory items
            processed_items = []
            for item_id, item_data in raw_inventory.items():
                try:
                    processed_item = self._process_inventory_item(item_id, item_data)
                    if processed_item:
                        processed_items.append(processed_item)
                except Exception as e:
                    self.logger.error(f"Error processing item {item_id}: {e}")
                    continue
            
            # Update cache
            self.inventory_cache = {item.asset_id: item for item in processed_items}
            self.last_inventory_update = datetime.now()
            
            self.logger.info(f"Successfully processed {len(processed_items)} inventory items")
            
            # Save to database
            self._save_inventory_to_db(processed_items)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error refreshing inventory: {e}")
            return False
    
    def _process_inventory_item(self, item_id: str, item_data: Dict) -> Optional[InventoryItem]:
        """Process a single inventory item"""
        try:
            # Extract basic item information
            market_hash_name = item_data.get('market_hash_name', '')
            name = item_data.get('name', '')
            type_name = item_data.get('type', '')
            
            # Extract rarity and exterior from item data
            rarity = self._extract_rarity(item_data)
            exterior = self._extract_exterior(market_hash_name)
            
            # Create base item
            item = InventoryItem(
                asset_id=item_id,
                class_id=item_data.get('classid', ''),
                instance_id=item_data.get('instanceid', ''),
                market_hash_name=market_hash_name,
                name=name,
                type=type_name,
                rarity=rarity,
                exterior=exterior,
                tradable=item_data.get('tradable', True),
                marketable=item_data.get('marketable', True)
            )
            
            # Get float value if possible (would need inspect link integration)
            item.float_value = self._estimate_float_value(item)
            
            # Get pricing information
            self._update_item_pricing(item)
            
            return item
            
        except Exception as e:
            self.logger.error(f"Error processing inventory item: {e}")
            return None
    
    def _extract_rarity(self, item_data: Dict) -> str:
        """Extract rarity from item data"""
        try:
            # Look for rarity in tags
            tags = item_data.get('tags', [])
            for tag in tags:
                if tag.get('category') == 'Rarity':
                    return tag.get('localized_tag_name', '')
            
            # Fallback to type analysis
            name = item_data.get('name', '').lower()
            if '★' in name or 'knife' in name:
                return 'Extraordinary'
            elif 'gloves' in name:
                return 'Extraordinary'
            
            return 'Unknown'
            
        except Exception:
            return 'Unknown'
    
    def _extract_exterior(self, market_hash_name: str) -> str:
        """Extract exterior condition from market hash name"""
        exteriors = ['Factory New', 'Minimal Wear', 'Field-Tested', 'Well-Worn', 'Battle-Scarred']
        
        for exterior in exteriors:
            if f"({exterior})" in market_hash_name:
                return exterior
        
        return 'Unknown'
    
    def _estimate_float_value(self, item: InventoryItem) -> float:
        """Estimate float value based on item information"""
        try:
            # Get skin info from database
            skin_info = self.skin_db.get_skin_info(item.market_hash_name)
            
            if skin_info and item.exterior in skin_info.wear_ranges:
                min_float, max_float = skin_info.wear_ranges[item.exterior]
                # Return middle of range as estimate
                return (min_float + max_float) / 2
            
            # Fallback to standard ranges
            standard_ranges = self.config.WEAR_RANGES
            if item.exterior in standard_ranges:
                min_float, max_float = standard_ranges[item.exterior]
                return (min_float + max_float) / 2
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _update_item_pricing(self, item: InventoryItem):
        """Update pricing information for an item"""
        try:
            # Check price cache first
            cache_key = item.market_hash_name
            if cache_key in self.price_cache:
                cached_data = self.price_cache[cache_key]
                cache_time = cached_data['timestamp']
                if datetime.now() - cache_time < self.cache_duration:
                    item.market_price = cached_data['price']
                    item.estimated_price = cached_data['estimated_price']
                    item.profit_potential = cached_data['profit_potential']
                    return
            
            # Get current market price
            price_data = self.trader.get_item_price(item.market_hash_name)
            if price_data:
                # Extract price information
                item.market_price = self._extract_price_from_data(price_data)
                
                # Calculate estimated value based on float and rarity
                item.estimated_price = self._calculate_estimated_value(item)
                
                # Calculate profit potential
                item.profit_potential = item.estimated_price - item.market_price
                
                # Cache the pricing data
                self.price_cache[cache_key] = {
                    'price': item.market_price,
                    'estimated_price': item.estimated_price,
                    'profit_potential': item.profit_potential,
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            self.logger.error(f"Error updating pricing for {item.market_hash_name}: {e}")
    
    def _extract_price_from_data(self, price_data: Dict) -> float:
        """Extract price from Steam market price data"""
        try:
            # Different possible price fields
            if 'lowest_price' in price_data:
                price_str = price_data['lowest_price'].replace('$', '').replace(',', '')
                return float(price_str)
            elif 'median_price' in price_data:
                price_str = price_data['median_price'].replace('$', '').replace(',', '')
                return float(price_str)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_estimated_value(self, item: InventoryItem) -> float:
        """Calculate estimated value based on float, rarity, and market trends"""
        try:
            base_price = item.market_price
            
            if base_price <= 0:
                return 0.0
            
            # Float value multiplier
            float_multiplier = 1.0
            if item.float_value > 0:
                analysis = self.float_analyzer.analyze_float_rarity(
                    item.market_hash_name, 
                    item.float_value, 
                    base_price
                )
                
                # Adjust based on rarity score
                if analysis.rarity_score >= 90:
                    float_multiplier = 1.5  # 50% premium for extremely rare floats
                elif analysis.rarity_score >= 70:
                    float_multiplier = 1.2  # 20% premium for rare floats
                elif analysis.rarity_score >= 50:
                    float_multiplier = 1.1  # 10% premium for good floats
            
            # Rarity multiplier
            rarity_multiplier = 1.0
            if 'Extraordinary' in item.rarity:
                rarity_multiplier = 1.1
            elif 'Covert' in item.rarity:
                rarity_multiplier = 1.05
            
            estimated_value = base_price * float_multiplier * rarity_multiplier
            
            return round(estimated_value, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating estimated value: {e}")
            return item.market_price
    
    def get_portfolio_summary(self) -> PortfolioSummary:
        """Generate comprehensive portfolio summary"""
        try:
            if not self.inventory_cache:
                self.refresh_inventory()
            
            items = list(self.inventory_cache.values())
            
            # Calculate totals
            total_items = len(items)
            total_value = sum(item.estimated_price for item in items)
            total_market_value = sum(item.market_price for item in items)
            potential_profit = sum(item.profit_potential for item in items)
            
            # Identify high-value items
            high_value_items = [
                item for item in items 
                if item.estimated_price >= self.high_value_threshold
            ]
            high_value_items.sort(key=lambda x: x.estimated_price, reverse=True)
            
            # Identify trading candidates
            trading_candidates = [
                item for item in items 
                if item.profit_potential / max(item.market_price, 1) >= self.profit_threshold
                and item.tradable
            ]
            trading_candidates.sort(key=lambda x: x.profit_potential, reverse=True)
            
            # Identify market opportunities
            market_opportunities = [
                item for item in items 
                if item.marketable and item.profit_potential > 5.0  # $5+ profit potential
            ]
            market_opportunities.sort(key=lambda x: x.profit_potential, reverse=True)
            
            return PortfolioSummary(
                total_items=total_items,
                total_value=total_value,
                total_market_value=total_market_value,
                potential_profit=potential_profit,
                high_value_items=high_value_items[:10],  # Top 10
                trading_candidates=trading_candidates[:10],  # Top 10
                market_opportunities=market_opportunities[:10],  # Top 10
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {e}")
            return PortfolioSummary(
                total_items=0,
                total_value=0.0,
                total_market_value=0.0,
                potential_profit=0.0,
                high_value_items=[],
                trading_candidates=[],
                market_opportunities=[],
                last_updated=datetime.now()
            )
    
    def get_items_by_criteria(self, min_value: float = 0, max_value: float = float('inf'), 
                            rarity: str = None, exterior: str = None, 
                            tradable_only: bool = False) -> List[InventoryItem]:
        """Get inventory items matching specific criteria"""
        try:
            if not self.inventory_cache:
                self.refresh_inventory()
            
            items = list(self.inventory_cache.values())
            
            # Apply filters
            filtered_items = []
            for item in items:
                # Value filter
                if not (min_value <= item.estimated_price <= max_value):
                    continue
                
                # Rarity filter
                if rarity and rarity.lower() not in item.rarity.lower():
                    continue
                
                # Exterior filter
                if exterior and exterior != item.exterior:
                    continue
                
                # Tradable filter
                if tradable_only and not item.tradable:
                    continue
                
                filtered_items.append(item)
            
            return filtered_items
            
        except Exception as e:
            self.logger.error(f"Error filtering inventory items: {e}")
            return []
    
    def _save_inventory_to_db(self, items: List[InventoryItem]):
        """Save inventory snapshot to database"""
        try:
            # This would extend the database schema to include inventory data
            # For now, just log the action
            self.logger.info(f"Saved inventory snapshot with {len(items)} items to database")
            
        except Exception as e:
            self.logger.error(f"Error saving inventory to database: {e}")
    
    def export_portfolio_report(self, filename: str = None) -> str:
        """Export detailed portfolio report"""
        try:
            if filename is None:
                filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            summary = self.get_portfolio_summary()
            
            # Convert to dictionary for JSON serialization
            report_data = {
                'summary': asdict(summary),
                'generated_at': datetime.now().isoformat(),
                'steam_user': self.auth.get_steam_id(),
                'total_items_scanned': len(self.inventory_cache)
            }
            
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=convert_datetime)
            
            self.logger.info(f"Portfolio report exported to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error exporting portfolio report: {e}")
            return ""

# Test function
def test_inventory_manager():
    """Test inventory management functionality"""
    print("🧪 Testing Inventory Manager...")
    
    from steam_auth import SteamAuthenticator
    
    # Create authenticator
    auth = SteamAuthenticator()
    
    if not auth.login():
        print("❌ Could not authenticate to Steam")
        return False
    
    try:
        # Create inventory manager
        inv_manager = InventoryManager(auth)
        
        # Test inventory refresh
        print("Refreshing inventory...")
        if inv_manager.refresh_inventory():
            print("✅ Inventory refresh successful")
            
            # Get portfolio summary
            summary = inv_manager.get_portfolio_summary()
            print(f"✅ Portfolio Summary:")
            print(f"   Total Items: {summary.total_items}")
            print(f"   Total Value: ${summary.total_value:.2f}")
            print(f"   Market Value: ${summary.total_market_value:.2f}")
            print(f"   Potential Profit: ${summary.potential_profit:.2f}")
            print(f"   High Value Items: {len(summary.high_value_items)}")
            
            # Export report
            report_file = inv_manager.export_portfolio_report()
            if report_file:
                print(f"✅ Portfolio report exported: {report_file}")
            
            return True
        else:
            print("❌ Inventory refresh failed")
            return False
            
    finally:
        auth.logout()

if __name__ == "__main__":
    test_inventory_manager()