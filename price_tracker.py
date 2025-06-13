import os
import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import pandas as pd

from steam_auth import SteamAuthenticator, SteamMarketTrader
from optimized_steam_api import OptimizedSteamAPI
from database import FloatDatabase
from config import FloatCheckerConfig

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class PriceChangeType(Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    STABLE = "stable"
    VOLATILE = "volatile"

class AlertType(Enum):
    PRICE_DROP = "price_drop"
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    ARBITRAGE = "arbitrage"
    FLOAT_OPPORTUNITY = "float_opportunity"

@dataclass
class PricePoint:
    timestamp: datetime
    price: float
    volume: int = 0
    lowest_price: float = 0.0
    median_price: float = 0.0
    source: str = "steam_market"

@dataclass
class PriceAlert:
    alert_id: str
    item_name: str
    alert_type: AlertType
    current_price: float
    threshold_price: float
    change_percentage: float
    message: str
    created_at: datetime
    triggered_at: datetime
    is_active: bool = True

@dataclass
class MarketTrend:
    item_name: str
    current_price: float
    price_24h_ago: float
    price_7d_ago: float
    change_24h: float
    change_7d: float
    change_type: PriceChangeType
    volatility_score: float
    volume_trend: str
    confidence_score: float
    last_updated: datetime

class PriceTracker:
    def __init__(self, authenticator: SteamAuthenticator = None):
        self.auth = authenticator
        self.trader = SteamMarketTrader(authenticator) if authenticator else None
        self.api = OptimizedSteamAPI()
        self.database = FloatDatabase()
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Price data storage
        self.price_history = defaultdict(lambda: deque(maxlen=1000))  # Store last 1000 points per item
        self.current_prices = {}
        self.price_alerts = {}
        self.watched_items = set()
        
        # Tracking settings
        self.update_interval = 300  # 5 minutes default
        self.alert_cooldown = 3600  # 1 hour cooldown per alert type
        self.min_price_change = 0.05  # 5% minimum change for alerts
        self.volatility_threshold = 0.3  # 30% volatility threshold
        
        # Alert handlers
        self.alert_handlers = []
        
        # Tracking state
        self.is_tracking = False
        self.tracking_thread = None
        
        # Load configuration
        self._load_tracking_config()
        self._load_watched_items()
    
    def _load_tracking_config(self):
        """Load price tracking configuration"""
        try:
            config_file = "price_tracking_config.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.update_interval = config.get('update_interval', 300)
                    self.min_price_change = config.get('min_price_change', 0.05)
                    self.volatility_threshold = config.get('volatility_threshold', 0.3)
                    self.logger.info("Price tracking configuration loaded")
        except Exception as e:
            self.logger.error(f"Error loading tracking config: {e}")
    
    def _load_watched_items(self):
        """Load list of items to track"""
        try:
            # Load from config file
            watchlist_file = "price_watchlist.json"
            if os.path.exists(watchlist_file):
                with open(watchlist_file, 'r') as f:
                    watchlist = json.load(f)
                    self.watched_items = set(watchlist.get('items', []))
            
            # Add high-value items from config if watchlist is empty
            if not self.watched_items:
                high_value_items = [
                    "AK-47 | Fire Serpent (Minimal Wear)",
                    "AWP | Dragon Lore (Factory New)",
                    "AWP | Dragon Lore (Minimal Wear)", 
                    "M4A4 | Howl (Factory New)",
                    "M4A4 | Howl (Minimal Wear)",
                    "Karambit | Fade (Factory New)",
                    "Butterfly Knife | Fade (Factory New)",
                    "Sport Gloves | Pandora's Box (Field-Tested)",
                    "AK-47 | Redline (Factory New)",
                    "AWP | Asiimov (Factory New)"
                ]
                self.watched_items = set(high_value_items)
                self.save_watchlist()
            
            self.logger.info(f"Loaded {len(self.watched_items)} items to track")
            
        except Exception as e:
            self.logger.error(f"Error loading watched items: {e}")
    
    def add_to_watchlist(self, item_names: List[str]):
        """Add items to price tracking watchlist"""
        try:
            added_count = 0
            for item_name in item_names:
                if item_name not in self.watched_items:
                    self.watched_items.add(item_name)
                    added_count += 1
            
            if added_count > 0:
                self.save_watchlist()
                self.logger.info(f"Added {added_count} items to watchlist")
            
            return added_count
            
        except Exception as e:
            self.logger.error(f"Error adding to watchlist: {e}")
            return 0
    
    def remove_from_watchlist(self, item_names: List[str]):
        """Remove items from watchlist"""
        try:
            removed_count = 0
            for item_name in item_names:
                if item_name in self.watched_items:
                    self.watched_items.remove(item_name)
                    removed_count += 1
            
            if removed_count > 0:
                self.save_watchlist()
                self.logger.info(f"Removed {removed_count} items from watchlist")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error removing from watchlist: {e}")
            return 0
    
    def save_watchlist(self):
        """Save current watchlist to file"""
        try:
            watchlist_data = {
                'items': list(self.watched_items),
                'last_updated': datetime.now().isoformat()
            }
            
            with open("price_watchlist.json", 'w') as f:
                json.dump(watchlist_data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error saving watchlist: {e}")
    
    def add_alert_handler(self, handler: Callable[[PriceAlert], None]):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    def create_price_alert(self, item_name: str, alert_type: AlertType, 
                          threshold: float, direction: str = "below") -> str:
        """Create a new price alert"""
        try:
            alert_id = f"{alert_type.value}_{item_name}_{int(time.time())}"
            
            # Get current price for reference
            current_price = self.get_current_price(item_name)
            
            alert = PriceAlert(
                alert_id=alert_id,
                item_name=item_name,
                alert_type=alert_type,
                current_price=current_price,
                threshold_price=threshold,
                change_percentage=0.0,
                message=f"Alert for {item_name}: {alert_type.value} at ${threshold:.2f}",
                created_at=datetime.now(),
                triggered_at=datetime.now(),
                is_active=True
            )
            
            self.price_alerts[alert_id] = alert
            
            # Add item to watchlist if not already there
            if item_name not in self.watched_items:
                self.add_to_watchlist([item_name])
            
            self.logger.info(f"Created price alert {alert_id} for {item_name}")
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Error creating price alert: {e}")
            return ""
    
    def start_tracking(self):
        """Start real-time price tracking"""
        if self.is_tracking:
            self.logger.warning("Price tracking already running")
            return
        
        try:
            self.is_tracking = True
            self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.tracking_thread.start()
            self.logger.info("Price tracking started")
            
        except Exception as e:
            self.logger.error(f"Error starting price tracking: {e}")
            self.is_tracking = False
    
    def stop_tracking(self):
        """Stop price tracking"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=10)
        self.logger.info("Price tracking stopped")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.is_tracking:
            try:
                # Update prices for all watched items
                asyncio.run(self._update_all_prices())
                
                # Check for alerts
                self._check_price_alerts()
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in tracking loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    async def _update_all_prices(self):
        """Update prices for all watched items"""
        try:
            if not self.watched_items:
                return
            
            self.logger.info(f"Updating prices for {len(self.watched_items)} items")
            
            # Use async API for efficient batch updates
            async with OptimizedSteamAPI() as api:
                tasks = []
                for item_name in self.watched_items:
                    task = self._update_item_price(api, item_name)
                    tasks.append(task)
                
                # Process in batches to avoid overwhelming the API
                batch_size = 5
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    await asyncio.gather(*batch, return_exceptions=True)
                    
                    # Small delay between batches
                    await asyncio.sleep(1)
            
            self.logger.info("Price update completed")
            
        except Exception as e:
            self.logger.error(f"Error updating prices: {e}")
    
    async def _update_item_price(self, api: OptimizedSteamAPI, item_name: str):
        """Update price for a single item"""
        try:
            # Get current market data
            market_data = await api.get_item_price(item_name)
            
            if market_data:
                # Extract price information
                price_point = self._extract_price_point(market_data)
                
                # Store price history
                self.price_history[item_name].append(price_point)
                self.current_prices[item_name] = price_point.price
                
                # Save to database
                self._save_price_to_db(item_name, price_point)
                
        except Exception as e:
            self.logger.error(f"Error updating price for {item_name}: {e}")
    
    def _extract_price_point(self, market_data: Dict) -> PricePoint:
        """Extract price point from market data"""
        try:
            # Extract prices from different fields
            lowest_price = 0.0
            median_price = 0.0
            
            if 'lowest_price' in market_data:
                price_str = market_data['lowest_price'].replace('$', '').replace(',', '')
                lowest_price = float(price_str)
            
            if 'median_price' in market_data:
                price_str = market_data['median_price'].replace('$', '').replace(',', '')
                median_price = float(price_str)
            
            # Use lowest price as primary, fallback to median
            primary_price = lowest_price if lowest_price > 0 else median_price
            
            # Extract volume if available
            volume = market_data.get('volume', 0)
            
            return PricePoint(
                timestamp=datetime.now(),
                price=primary_price,
                volume=volume,
                lowest_price=lowest_price,
                median_price=median_price,
                source="steam_market"
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting price point: {e}")
            return PricePoint(
                timestamp=datetime.now(),
                price=0.0,
                source="error"
            )
    
    def _check_price_alerts(self):
        """Check all active price alerts"""
        try:
            for alert_id, alert in self.price_alerts.items():
                if not alert.is_active:
                    continue
                
                current_price = self.current_prices.get(alert.item_name, 0)
                if current_price <= 0:
                    continue
                
                # Check if alert conditions are met
                if self._should_trigger_alert(alert, current_price):
                    self._trigger_alert(alert, current_price)
                    
        except Exception as e:
            self.logger.error(f"Error checking price alerts: {e}")
    
    def _should_trigger_alert(self, alert: PriceAlert, current_price: float) -> bool:
        """Check if alert should be triggered"""
        try:
            if alert.alert_type == AlertType.PRICE_DROP:
                return current_price <= alert.threshold_price
            
            elif alert.alert_type == AlertType.PRICE_SPIKE:
                return current_price >= alert.threshold_price
            
            elif alert.alert_type == AlertType.VOLUME_SPIKE:
                # Would need volume comparison logic
                return False
            
            elif alert.alert_type == AlertType.ARBITRAGE:
                # Would need cross-platform price comparison
                return False
            
            return False
            
        except Exception:
            return False
    
    def _trigger_alert(self, alert: PriceAlert, current_price: float):
        """Trigger a price alert"""
        try:
            # Calculate change percentage
            if alert.current_price > 0:
                change_pct = (current_price - alert.current_price) / alert.current_price
            else:
                change_pct = 0.0
            
            # Update alert
            alert.triggered_at = datetime.now()
            alert.current_price = current_price
            alert.change_percentage = change_pct
            alert.message = self._generate_alert_message(alert, current_price, change_pct)
            
            # Call alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert handler: {e}")
            
            # Deactivate alert (single-use)
            alert.is_active = False
            
            self.logger.info(f"Alert triggered: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    def _generate_alert_message(self, alert: PriceAlert, current_price: float, change_pct: float) -> str:
        """Generate alert message"""
        try:
            direction = "increased" if change_pct > 0 else "decreased"
            
            message = (f"🚨 {alert.alert_type.value.upper()}: {alert.item_name}\n"
                      f"Price {direction} to ${current_price:.2f} "
                      f"({change_pct:+.1%})\n"
                      f"Threshold: ${alert.threshold_price:.2f}")
            
            return message
            
        except Exception:
            return f"Alert triggered for {alert.item_name}"
    
    def get_current_price(self, item_name: str) -> float:
        """Get current price for an item"""
        return self.current_prices.get(item_name, 0.0)
    
    def get_price_history(self, item_name: str, hours: int = 24) -> List[PricePoint]:
        """Get price history for an item"""
        try:
            if item_name not in self.price_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            history = []
            for point in self.price_history[item_name]:
                if point.timestamp >= cutoff_time:
                    history.append(point)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting price history: {e}")
            return []
    
    def calculate_market_trend(self, item_name: str) -> Optional[MarketTrend]:
        """Calculate market trend for an item"""
        try:
            history = list(self.price_history[item_name])
            if len(history) < 2:
                return None
            
            current_price = history[-1].price
            
            # Find prices at different time intervals
            now = datetime.now()
            price_24h = self._get_price_at_time(history, now - timedelta(hours=24))
            price_7d = self._get_price_at_time(history, now - timedelta(days=7))
            
            # Calculate changes
            change_24h = ((current_price - price_24h) / price_24h) if price_24h > 0 else 0
            change_7d = ((current_price - price_7d) / price_7d) if price_7d > 0 else 0
            
            # Determine change type
            if abs(change_24h) >= 0.3:  # 30% change
                change_type = PriceChangeType.VOLATILE
            elif change_24h >= 0.05:  # 5% increase
                change_type = PriceChangeType.INCREASE
            elif change_24h <= -0.05:  # 5% decrease
                change_type = PriceChangeType.DECREASE
            else:
                change_type = PriceChangeType.STABLE
            
            # Calculate volatility score
            volatility = self._calculate_volatility(history)
            
            # Calculate confidence score
            confidence = min(len(history) / 100, 1.0)  # More data = higher confidence
            
            return MarketTrend(
                item_name=item_name,
                current_price=current_price,
                price_24h_ago=price_24h,
                price_7d_ago=price_7d,
                change_24h=change_24h,
                change_7d=change_7d,
                change_type=change_type,
                volatility_score=volatility,
                volume_trend="unknown",  # Would need volume analysis
                confidence_score=confidence,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating market trend: {e}")
            return None
    
    def _get_price_at_time(self, history: List[PricePoint], target_time: datetime) -> float:
        """Get price closest to target time"""
        try:
            if not history:
                return 0.0
            
            # Find closest price point
            closest_point = min(history, key=lambda p: abs((p.timestamp - target_time).total_seconds()))
            return closest_point.price
            
        except Exception:
            return 0.0
    
    def _calculate_volatility(self, history: List[PricePoint]) -> float:
        """Calculate price volatility score"""
        try:
            if len(history) < 3:
                return 0.0
            
            prices = [p.price for p in history if p.price > 0]
            if len(prices) < 3:
                return 0.0
            
            # Calculate standard deviation
            mean_price = sum(prices) / len(prices)
            variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
            std_dev = variance ** 0.5
            
            # Normalize by mean price
            volatility = std_dev / mean_price if mean_price > 0 else 0
            
            return min(volatility, 1.0)  # Cap at 100%
            
        except Exception:
            return 0.0
    
    def _save_price_to_db(self, item_name: str, price_point: PricePoint):
        """Save price point to database"""
        try:
            # This would extend the database schema to include price history
            pass
        except Exception as e:
            self.logger.error(f"Error saving price to database: {e}")
    
    def export_price_data(self, item_name: str = None, filename: str = None) -> str:
        """Export price data to CSV"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"price_data_{timestamp}.csv"
            
            # Prepare data for export
            export_data = []
            
            if item_name:
                # Export single item
                items_to_export = [item_name] if item_name in self.price_history else []
            else:
                # Export all items
                items_to_export = list(self.price_history.keys())
            
            for item in items_to_export:
                for point in self.price_history[item]:
                    export_data.append({
                        'item_name': item,
                        'timestamp': point.timestamp.isoformat(),
                        'price': point.price,
                        'volume': point.volume,
                        'lowest_price': point.lowest_price,
                        'median_price': point.median_price,
                        'source': point.source
                    })
            
            # Create DataFrame and save
            if export_data:
                df = pd.DataFrame(export_data)
                df.to_csv(filename, index=False)
                self.logger.info(f"Price data exported to {filename}")
                return filename
            else:
                self.logger.warning("No price data to export")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error exporting price data: {e}")
            return ""
    
    def get_tracking_summary(self) -> Dict:
        """Get summary of current tracking status"""
        try:
            summary = {
                'is_tracking': self.is_tracking,
                'watched_items_count': len(self.watched_items),
                'active_alerts_count': sum(1 for alert in self.price_alerts.values() if alert.is_active),
                'items_with_prices': len(self.current_prices),
                'update_interval': self.update_interval,
                'last_update': max([
                    max(history, key=lambda p: p.timestamp).timestamp
                    for history in self.price_history.values()
                    if history
                ], default=None)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting tracking summary: {e}")
            return {}

# Telegram alert handler
def telegram_alert_handler(alert: PriceAlert):
    """Send alert via Telegram"""
    try:
        from config import FloatCheckerConfig
        config = FloatCheckerConfig()
        
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            return
        
        if not REQUESTS_AVAILABLE:
            return
        
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        
        data = {
            'chat_id': config.TELEGRAM_CHAT_ID,
            'text': alert.message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            logging.info("Telegram alert sent successfully")
        else:
            logging.error(f"Failed to send Telegram alert: {response.status_code}")
            
    except Exception as e:
        logging.error(f"Error sending Telegram alert: {e}")

# Test function
def test_price_tracker():
    """Test price tracking functionality"""
    print("🧪 Testing Price Tracker...")
    
    try:
        # Create price tracker
        tracker = PriceTracker()
        
        # Add test items to watchlist
        test_items = ["AK-47 | Redline (Field-Tested)", "AWP | Asiimov (Field-Tested)"]
        added = tracker.add_to_watchlist(test_items)
        print(f"✅ Added {added} items to watchlist")
        
        # Create test alert
        alert_id = tracker.create_price_alert(
            "AK-47 | Redline (Field-Tested)",
            AlertType.PRICE_DROP,
            10.0
        )
        
        if alert_id:
            print(f"✅ Created price alert: {alert_id}")
        
        # Add Telegram alert handler
        tracker.add_alert_handler(telegram_alert_handler)
        
        # Test current price retrieval
        current_price = tracker.get_current_price("AK-47 | Redline (Field-Tested)")
        print(f"Current price retrieved: ${current_price:.2f}")
        
        # Get tracking summary
        summary = tracker.get_tracking_summary()
        print("✅ Tracking Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print("✅ Price tracker test completed")
        print("Note: Start tracking with tracker.start_tracking() for real-time monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Price tracker test failed: {e}")
        return False

if __name__ == "__main__":
    test_price_tracker()