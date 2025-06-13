#!/usr/bin/env python3

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from decimal import Decimal, ROUND_HALF_UP

from steampy.client import SteamClient
from steampy.models import Asset, TradeOffer, Currency
from steampy.exceptions import InvalidCredentials, TooManyRequests, ApiException

from config import FloatCheckerConfig
from database import FloatDatabase
from float_analyzer import FloatAnalysis
from smart_scanner import SmartScanner
from csfloat_api import CSFloatAPI, FloatData

class TradeType(Enum):
    BUY = "buy"
    SELL = "sell"
    ARBITRAGE = "arbitrage"

class TradeStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TradeOpportunity:
    """Represents a profitable trading opportunity"""
    item_name: str
    market_price: float
    target_price: float
    profit_margin: float
    profit_percentage: float
    trade_type: TradeType
    risk_level: RiskLevel
    float_value: Optional[float] = None
    inspect_link: Optional[str] = None
    market_hash_name: str = ""
    estimated_sell_time: int = 24  # hours
    confidence_score: float = 0.0
    steam_market_url: str = ""
    
    def __post_init__(self):
        if not self.market_hash_name:
            self.market_hash_name = self.item_name
        
        # Calculate confidence score based on profit and risk
        base_confidence = min(self.profit_percentage * 10, 100)
        risk_modifier = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 0.3
        }
        self.confidence_score = base_confidence * risk_modifier[self.risk_level]

@dataclass
class TradeRecord:
    """Record of executed trade"""
    trade_id: str
    opportunity: TradeOpportunity
    status: TradeStatus
    created_at: datetime
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_profit: Optional[float] = None
    notes: str = ""
    steam_trade_id: Optional[str] = None

@dataclass
class PortfolioItem:
    """Item in trading portfolio"""
    item_name: str
    market_hash_name: str
    asset_id: str
    purchase_price: float
    current_market_price: float
    float_value: Optional[float] = None
    purchase_date: datetime = None
    target_sell_price: float = 0.0
    days_held: int = 0
    
    def __post_init__(self):
        if self.purchase_date is None:
            self.purchase_date = datetime.now()
        self.days_held = (datetime.now() - self.purchase_date).days
        
        # Set target sell price if not specified
        if self.target_sell_price == 0.0:
            self.target_sell_price = self.purchase_price * 1.15  # 15% minimum profit
    
    @property
    def unrealized_profit(self) -> float:
        return self.current_market_price - self.purchase_price
    
    @property
    def unrealized_profit_percentage(self) -> float:
        if self.purchase_price > 0:
            return (self.unrealized_profit / self.purchase_price) * 100
        return 0.0

class RiskAssessment:
    """Advanced risk assessment for trading decisions"""
    
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Risk thresholds
        self.risk_thresholds = {
            'min_profit_margin': 10.0,  # 10% minimum profit
            'max_investment_per_item': 500.0,  # $500 max per item
            'max_portfolio_percentage': 20.0,  # 20% max of portfolio in one item
            'min_volume_24h': 5,  # Minimum 5 sales in 24h
            'max_hold_time_days': 30,  # Maximum 30 days hold time
            'blacklisted_items': set()  # Items to never trade
        }
        
        # Market volatility tracking
        self.volatility_data = {}
        self.price_history = {}
    
    def assess_trade_risk(self, opportunity: TradeOpportunity, 
                         portfolio_value: float, current_holdings: List[PortfolioItem]) -> RiskLevel:
        """Comprehensive risk assessment for trade opportunity"""
        try:
            risk_factors = []
            
            # 1. Profit margin risk
            if opportunity.profit_percentage < self.risk_thresholds['min_profit_margin']:
                risk_factors.append('low_profit_margin')
            
            # 2. Investment size risk
            if opportunity.market_price > self.risk_thresholds['max_investment_per_item']:
                risk_factors.append('high_investment')
            
            # 3. Portfolio concentration risk
            item_concentration = self._calculate_portfolio_concentration(
                opportunity.item_name, opportunity.market_price, current_holdings, portfolio_value
            )
            if item_concentration > self.risk_thresholds['max_portfolio_percentage']:
                risk_factors.append('high_concentration')
            
            # 4. Market liquidity risk
            liquidity_score = self._assess_market_liquidity(opportunity.item_name)
            if liquidity_score < 0.3:
                risk_factors.append('low_liquidity')
            
            # 5. Price volatility risk
            volatility_score = self._assess_price_volatility(opportunity.item_name)
            if volatility_score > 0.7:
                risk_factors.append('high_volatility')
            
            # 6. Float value risk (for float-based trades)
            if opportunity.float_value:
                float_risk = self._assess_float_risk(opportunity.item_name, opportunity.float_value)
                if float_risk > 0.6:
                    risk_factors.append('float_risk')
            
            # 7. Blacklist check
            if opportunity.item_name in self.risk_thresholds['blacklisted_items']:
                risk_factors.append('blacklisted')
            
            # Calculate overall risk level
            risk_score = len(risk_factors) / 7.0  # Normalize to 0-1
            
            if risk_score >= 0.7 or 'blacklisted' in risk_factors:
                return RiskLevel.CRITICAL
            elif risk_score >= 0.5:
                return RiskLevel.HIGH
            elif risk_score >= 0.3:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            return RiskLevel.HIGH  # Default to high risk on error
    
    def _calculate_portfolio_concentration(self, item_name: str, item_value: float, 
                                         holdings: List[PortfolioItem], total_value: float) -> float:
        """Calculate what percentage of portfolio this item would represent"""
        current_item_value = sum(
            item.current_market_price for item in holdings 
            if item.item_name == item_name
        )
        total_item_value = current_item_value + item_value
        
        if total_value > 0:
            return (total_item_value / total_value) * 100
        return 0.0
    
    def _assess_market_liquidity(self, item_name: str) -> float:
        """Assess market liquidity for an item (0.0 = illiquid, 1.0 = highly liquid)"""
        # Simulate liquidity assessment based on item characteristics
        liquidity_score = 0.5  # Default medium liquidity
        
        # High liquidity items
        high_liquidity_keywords = ['AK-47', 'AWP', 'M4A4', 'M4A1-S', 'Glock', 'USP']
        if any(keyword in item_name for keyword in high_liquidity_keywords):
            liquidity_score += 0.3
        
        # Low liquidity items
        low_liquidity_keywords = ['StatTrak', 'Souvenir', 'Gloves', 'Music Kit']
        if any(keyword in item_name for keyword in low_liquidity_keywords):
            liquidity_score -= 0.2
        
        # Knife/high-value items have medium liquidity
        if any(knife in item_name for knife in ['Karambit', 'Butterfly', 'Bayonet']):
            liquidity_score = 0.6
        
        return max(0.0, min(1.0, liquidity_score))
    
    def _assess_price_volatility(self, item_name: str) -> float:
        """Assess price volatility for an item (0.0 = stable, 1.0 = highly volatile)"""
        # Simulate volatility assessment
        volatility_score = 0.3  # Default low volatility
        
        # High volatility items
        if any(keyword in item_name for keyword in ['Case Hardened', 'Fade', 'Marble Fade']):
            volatility_score += 0.4
        
        # Stable items
        if any(keyword in item_name for keyword in ['Redline', 'Asiimov', 'Hyper Beast']):
            volatility_score -= 0.1
        
        return max(0.0, min(1.0, volatility_score))
    
    def _assess_float_risk(self, item_name: str, float_value: float) -> float:
        """Assess risk associated with float value trading"""
        # Very low or very high floats have higher risk but higher reward
        if float_value <= 0.001 or float_value >= 0.999:
            return 0.8  # High risk but potentially high reward
        elif float_value <= 0.01 or float_value >= 0.95:
            return 0.4  # Medium risk
        else:
            return 0.2  # Low risk for normal floats

class TradingSystem:
    """Advanced automated trading system for CS2 items"""
    
    def __init__(self, username: str, password: str, shared_secret: str, identity_secret: str):
        self.config = FloatCheckerConfig()
        self.database = FloatDatabase()
        self.smart_scanner = SmartScanner()
        self.risk_assessor = RiskAssessment()
        self.setup_logging()
        
        # Steam client setup
        self.steam_client = None
        self.credentials = {
            'username': username,
            'password': password,
            'shared_secret': shared_secret,
            'identity_secret': identity_secret
        }
        
        # Trading state
        self.is_running = False
        self.portfolio: List[PortfolioItem] = []
        self.active_trades: List[TradeRecord] = []
        self.trade_history: List[TradeRecord] = []
        self.opportunities_queue = asyncio.Queue()
        
        # Performance tracking
        self.performance_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit': 0.0,
            'total_invested': 0.0,
            'win_rate': 0.0,
            'average_profit_per_trade': 0.0,
            'portfolio_value': 0.0,
            'session_start': datetime.now()
        }
        
        # Safety limits
        self.safety_limits = {
            'max_daily_trades': 20,
            'max_concurrent_trades': 5,
            'max_daily_investment': 1000.0,
            'emergency_stop_loss': -500.0  # Stop if losing more than $500
        }
        
        self.daily_trade_count = 0
        self.daily_investment = 0.0
        self.last_reset_date = datetime.now().date()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize the trading system"""
        try:
            self.logger.info("🔧 Initializing Trading System...")
            
            # Initialize Steam client
            if not await self._initialize_steam_client():
                return False
            
            # Load existing portfolio
            await self._load_portfolio()
            
            # Load trade history
            await self._load_trade_history()
            
            # Update portfolio values
            await self._update_portfolio_values()
            
            self.logger.info("✅ Trading System initialized successfully")
            self.logger.info(f"💰 Current Portfolio Value: ${self.performance_stats['portfolio_value']:.2f}")
            self.logger.info(f"📊 Active Trades: {len(self.active_trades)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trading system: {e}")
            return False
    
    async def _initialize_steam_client(self) -> bool:
        """Initialize and login to Steam"""
        try:
            self.steam_client = SteamClient(self.config.STEAM_API_KEY)
            
            # Login with credentials
            login_result = self.steam_client.login(
                username=self.credentials['username'],
                password=self.credentials['password'],
                steam_guard=self.credentials['shared_secret']
            )
            
            if login_result:
                self.logger.info("✅ Successfully logged into Steam")
                return True
            else:
                self.logger.error("❌ Failed to login to Steam")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Steam client initialization failed: {e}")
            return False
    
    async def start_trading(self):
        """Start the automated trading system"""
        if not self.steam_client:
            self.logger.error("❌ Steam client not initialized")
            return
        
        self.is_running = True
        self.logger.info("🚀 Starting automated trading system...")
        
        # Start concurrent tasks
        tasks = [
            self._opportunity_scanner(),
            self._trade_executor(),
            self._portfolio_monitor(),
            self._safety_monitor()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading system stopped by user")
        finally:
            await self.stop_trading()
    
    async def stop_trading(self):
        """Stop the trading system safely"""
        self.is_running = False
        self.logger.info("🛑 Stopping trading system...")
        
        # Save current state
        await self._save_portfolio()
        await self._save_trade_history()
        
        # Logout from Steam
        if self.steam_client:
            try:
                self.steam_client.logout()
                self.logger.info("✅ Logged out from Steam")
            except Exception as e:
                self.logger.error(f"Error during Steam logout: {e}")
    
    async def _opportunity_scanner(self):
        """Continuously scan for trading opportunities"""
        while self.is_running:
            try:
                self.logger.info("🔍 Scanning for trading opportunities...")
                
                # Get high-priority items from smart scanner
                scan_targets = self.smart_scanner.get_next_scan_targets(max_targets=20)
                
                for target in scan_targets:
                    if not self.is_running:
                        break
                    
                    # Check for arbitrage opportunities
                    opportunities = await self._find_arbitrage_opportunities(target.item_name)
                    
                    for opportunity in opportunities:
                        # Assess risk
                        risk_level = self.risk_assessor.assess_trade_risk(
                            opportunity, 
                            self.performance_stats['portfolio_value'], 
                            self.portfolio
                        )
                        opportunity.risk_level = risk_level
                        
                        # Only queue profitable, low-to-medium risk opportunities
                        if (opportunity.profit_percentage >= 10.0 and 
                            risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]):
                            
                            await self.opportunities_queue.put(opportunity)
                            self.logger.info(f"💡 Found opportunity: {opportunity.item_name} "
                                           f"(+{opportunity.profit_percentage:.1f}%, Risk: {risk_level.value})")
                
                # Brief pause before next scan
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in opportunity scanner: {e}")
                await asyncio.sleep(60)
    
    async def _find_arbitrage_opportunities(self, item_name: str) -> List[TradeOpportunity]:
        """Find arbitrage opportunities for a specific item"""
        opportunities = []
        
        try:
            # Simulate finding price differences (in real implementation, 
            # this would query multiple marketplaces)
            
            # Mock data for demonstration
            steam_market_price = 100.0  # Steam Community Market price
            third_party_price = 85.0    # Third-party site price
            
            if steam_market_price > third_party_price * 1.1:  # 10%+ profit margin
                profit_margin = steam_market_price - third_party_price
                profit_percentage = (profit_margin / third_party_price) * 100
                
                opportunity = TradeOpportunity(
                    item_name=item_name,
                    market_price=third_party_price,
                    target_price=steam_market_price,
                    profit_margin=profit_margin,
                    profit_percentage=profit_percentage,
                    trade_type=TradeType.ARBITRAGE,
                    risk_level=RiskLevel.MEDIUM  # Will be updated by risk assessor
                )
                
                opportunities.append(opportunity)
        
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunities for {item_name}: {e}")
        
        return opportunities
    
    async def _trade_executor(self):
        """Execute trades from the opportunities queue"""
        while self.is_running:
            try:
                # Check safety limits
                if not self._check_safety_limits():
                    await asyncio.sleep(300)  # Wait 5 minutes if limits exceeded
                    continue
                
                # Get next opportunity
                try:
                    opportunity = await asyncio.wait_for(
                        self.opportunities_queue.get(), timeout=60
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Execute the trade
                success = await self._execute_trade(opportunity)
                
                if success:
                    self.logger.info(f"✅ Successfully executed trade for {opportunity.item_name}")
                    self.daily_trade_count += 1
                    self.daily_investment += opportunity.market_price
                else:
                    self.logger.warning(f"❌ Failed to execute trade for {opportunity.item_name}")
                
                # Brief pause between trades
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in trade executor: {e}")
                await asyncio.sleep(30)
    
    async def _execute_trade(self, opportunity: TradeOpportunity) -> bool:
        """Execute a specific trade"""
        try:
            # Create trade record
            trade_record = TradeRecord(
                trade_id=f"trade_{int(time.time())}",
                opportunity=opportunity,
                status=TradeStatus.PENDING,
                created_at=datetime.now()
            )
            
            # Add to active trades
            self.active_trades.append(trade_record)
            
            # In a real implementation, this would:
            # 1. Send trade offer via Steam API
            # 2. Wait for acceptance
            # 3. Update portfolio
            # 4. Track performance
            
            # Simulate trade execution
            await asyncio.sleep(2)  # Simulate network delay
            
            # Simulate successful trade (90% success rate)
            import random
            if random.random() < 0.9:
                trade_record.status = TradeStatus.ACCEPTED
                trade_record.executed_at = datetime.now()
                
                # Add to portfolio
                portfolio_item = PortfolioItem(
                    item_name=opportunity.item_name,
                    market_hash_name=opportunity.market_hash_name,
                    asset_id=f"asset_{int(time.time())}",
                    purchase_price=opportunity.market_price,
                    current_market_price=opportunity.target_price,
                    float_value=opportunity.float_value
                )
                
                self.portfolio.append(portfolio_item)
                
                # Update performance stats
                self.performance_stats['total_trades'] += 1
                self.performance_stats['successful_trades'] += 1
                self.performance_stats['total_invested'] += opportunity.market_price
                
                # Move to trade history
                self.active_trades.remove(trade_record)
                self.trade_history.append(trade_record)
                
                return True
            else:
                trade_record.status = TradeStatus.DECLINED
                self.active_trades.remove(trade_record)
                self.trade_history.append(trade_record)
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    def _check_safety_limits(self) -> bool:
        """Check if trading should continue based on safety limits"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trade_count = 0
            self.daily_investment = 0.0
            self.last_reset_date = today
        
        # Check daily limits
        if self.daily_trade_count >= self.safety_limits['max_daily_trades']:
            self.logger.warning("⚠️ Daily trade limit reached")
            return False
        
        if len(self.active_trades) >= self.safety_limits['max_concurrent_trades']:
            self.logger.warning("⚠️ Concurrent trade limit reached")
            return False
        
        if self.daily_investment >= self.safety_limits['max_daily_investment']:
            self.logger.warning("⚠️ Daily investment limit reached")
            return False
        
        # Check emergency stop loss
        total_profit = sum(item.unrealized_profit for item in self.portfolio)
        if total_profit <= self.safety_limits['emergency_stop_loss']:
            self.logger.error("🚨 EMERGENCY STOP LOSS TRIGGERED")
            self.is_running = False
            return False
        
        return True
    
    async def _portfolio_monitor(self):
        """Monitor portfolio performance and execute sell orders"""
        while self.is_running:
            try:
                await self._update_portfolio_values()
                
                # Check for items to sell
                for item in self.portfolio[:]:  # Copy list to allow modification
                    if self._should_sell_item(item):
                        await self._sell_item(item)
                
                # Update performance stats
                self._update_performance_stats()
                
                # Log portfolio status every 10 minutes
                await asyncio.sleep(600)
                self._log_portfolio_status()
                
            except Exception as e:
                self.logger.error(f"Error in portfolio monitor: {e}")
                await asyncio.sleep(300)
    
    def _should_sell_item(self, item: PortfolioItem) -> bool:
        """Determine if an item should be sold"""
        # Sell if target price reached
        if item.current_market_price >= item.target_sell_price:
            return True
        
        # Sell if held too long (30 days)
        if item.days_held >= 30:
            return True
        
        # Sell if losing more than 20%
        if item.unrealized_profit_percentage <= -20.0:
            return True
        
        return False
    
    async def _sell_item(self, item: PortfolioItem):
        """Sell an item from portfolio"""
        try:
            self.logger.info(f"💰 Selling {item.item_name} for ${item.current_market_price:.2f}")
            
            # In real implementation, this would create Steam market listing
            # Simulate sale
            await asyncio.sleep(1)
            
            # Update performance
            actual_profit = item.current_market_price - item.purchase_price
            self.performance_stats['total_profit'] += actual_profit
            
            # Remove from portfolio
            self.portfolio.remove(item)
            
            self.logger.info(f"✅ Sold {item.item_name} for ${actual_profit:.2f} profit")
            
        except Exception as e:
            self.logger.error(f"Error selling {item.item_name}: {e}")
    
    async def _safety_monitor(self):
        """Monitor system safety and health"""
        while self.is_running:
            try:
                # Check system health
                if not self.steam_client or not self.steam_client.is_session_alive():
                    self.logger.warning("⚠️ Steam session expired, attempting reconnection...")
                    await self._initialize_steam_client()
                
                # Monitor for unusual activity
                if len(self.active_trades) > 10:
                    self.logger.warning("⚠️ Unusual number of active trades detected")
                
                # Check portfolio concentration
                total_value = sum(item.current_market_price for item in self.portfolio)
                for item in self.portfolio:
                    concentration = (item.current_market_price / total_value) * 100 if total_value > 0 else 0
                    if concentration > 30:  # More than 30% in one item
                        self.logger.warning(f"⚠️ High concentration in {item.item_name}: {concentration:.1f}%")
                
                await asyncio.sleep(180)  # Check every 3 minutes
                
            except Exception as e:
                self.logger.error(f"Error in safety monitor: {e}")
                await asyncio.sleep(60)
    
    async def _update_portfolio_values(self):
        """Update current market values for all portfolio items"""
        for item in self.portfolio:
            # In real implementation, this would fetch current market prices
            # Simulate price updates with small random changes
            import random
            change_factor = random.uniform(0.95, 1.05)  # ±5% change
            item.current_market_price *= change_factor
    
    def _update_performance_stats(self):
        """Update comprehensive performance statistics"""
        total_invested = sum(item.purchase_price for item in self.portfolio)
        total_current_value = sum(item.current_market_price for item in self.portfolio)
        
        self.performance_stats.update({
            'portfolio_value': total_current_value,
            'total_invested': total_invested,
            'win_rate': (self.performance_stats['successful_trades'] / 
                        max(self.performance_stats['total_trades'], 1)) * 100,
            'average_profit_per_trade': (self.performance_stats['total_profit'] / 
                                       max(self.performance_stats['successful_trades'], 1))
        })
    
    def _log_portfolio_status(self):
        """Log current portfolio status"""
        self.logger.info("📊 PORTFOLIO STATUS")
        self.logger.info(f"💰 Total Value: ${self.performance_stats['portfolio_value']:.2f}")
        self.logger.info(f"📈 Total P&L: ${self.performance_stats['total_profit']:.2f}")
        self.logger.info(f"🎯 Win Rate: {self.performance_stats['win_rate']:.1f}%")
        self.logger.info(f"📊 Items Held: {len(self.portfolio)}")
        self.logger.info(f"🔄 Active Trades: {len(self.active_trades)}")
    
    async def _load_portfolio(self):
        """Load portfolio from storage"""
        try:
            # In real implementation, load from database
            self.portfolio = []
            self.logger.info("✅ Portfolio loaded")
        except Exception as e:
            self.logger.error(f"Error loading portfolio: {e}")
    
    async def _save_portfolio(self):
        """Save portfolio to storage"""
        try:
            # In real implementation, save to database
            self.logger.info("✅ Portfolio saved")
        except Exception as e:
            self.logger.error(f"Error saving portfolio: {e}")
    
    async def _load_trade_history(self):
        """Load trade history from storage"""
        try:
            self.trade_history = []
            self.logger.info("✅ Trade history loaded")
        except Exception as e:
            self.logger.error(f"Error loading trade history: {e}")
    
    async def _save_trade_history(self):
        """Save trade history to storage"""
        try:
            self.logger.info("✅ Trade history saved")
        except Exception as e:
            self.logger.error(f"Error saving trade history: {e}")

# Test function
async def test_trading_system():
    """Test trading system functionality"""
    print("🧪 Testing Trading System...")
    
    # Mock credentials for testing
    trading_system = TradingSystem(
        username="test_user",
        password="test_pass", 
        shared_secret="test_secret",
        identity_secret="test_identity"
    )
    
    # Test risk assessment
    print("Test 1: Risk Assessment")
    risk_assessor = RiskAssessment()
    
    test_opportunity = TradeOpportunity(
        item_name="AK-47 | Redline",
        market_price=50.0,
        target_price=60.0,
        profit_margin=10.0,
        profit_percentage=20.0,
        trade_type=TradeType.ARBITRAGE,
        risk_level=RiskLevel.LOW
    )
    
    risk_level = risk_assessor.assess_trade_risk(test_opportunity, 1000.0, [])
    print(f"✅ Risk assessment completed: {risk_level.value}")
    
    # Test portfolio item
    print("Test 2: Portfolio Item")
    portfolio_item = PortfolioItem(
        item_name="AWP | Dragon Lore",
        market_hash_name="AWP | Dragon Lore (Field-Tested)",
        asset_id="12345",
        purchase_price=2000.0,
        current_market_price=2200.0
    )
    
    print(f"✅ Portfolio item created: ${portfolio_item.unrealized_profit:.2f} profit "
          f"({portfolio_item.unrealized_profit_percentage:.1f}%)")
    
    print("✅ Trading System test completed successfully")

if __name__ == "__main__":
    asyncio.run(test_trading_system())