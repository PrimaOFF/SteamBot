#!/usr/bin/env python3
import os
import asyncio
import logging
import json
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sys

# Import all our trading system components
from steam_auth import SteamAuthenticator, SteamMarketTrader
from inventory_manager import InventoryManager, InventoryItem, PortfolioSummary
from price_tracker import PriceTracker, PriceAlert, AlertType, telegram_alert_handler
from trade_evaluator import TradeOfferEvaluator, TradeEvaluation, TradeResult
from opportunity_detector import OpportunityDetector, MarketOpportunity, OpportunityType, OpportunityPriority
from float_analyzer import FloatAnalyzer
from database import FloatDatabase
from config import FloatCheckerConfig

class DashboardMode(Enum):
    OVERVIEW = "overview"
    PORTFOLIO = "portfolio"
    OPPORTUNITIES = "opportunities"
    PRICE_TRACKING = "price_tracking"
    TRADE_EVALUATION = "trade_evaluation"
    SETTINGS = "settings"

@dataclass
class DashboardStats:
    total_portfolio_value: float
    active_opportunities: int
    high_priority_opportunities: int
    active_price_alerts: int
    items_tracked: int
    recent_trades_evaluated: int
    system_uptime: str
    last_scan_time: Optional[datetime]
    profit_potential: float

class TradingDashboard:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = FloatCheckerConfig()
        
        # Core components
        self.authenticator = None
        self.inventory_manager = None
        self.price_tracker = None
        self.trade_evaluator = None
        self.opportunity_detector = None
        self.float_analyzer = None
        
        # Dashboard state
        self.is_authenticated = False
        self.current_mode = DashboardMode.OVERVIEW
        self.start_time = datetime.now()
        self.auto_mode = False
        
        # Statistics
        self.stats = DashboardStats(
            total_portfolio_value=0.0,
            active_opportunities=0,
            high_priority_opportunities=0,
            active_price_alerts=0,
            items_tracked=0,
            recent_trades_evaluated=0,
            system_uptime="0m",
            last_scan_time=None,
            profit_potential=0.0
        )
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create logs directory if it doesn't exist
            if not os.path.exists('logs'):
                os.makedirs('logs')
            
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(f'logs/trading_dashboard_{datetime.now().strftime("%Y%m%d")}.log'),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def start(self):
        """Start the trading dashboard"""
        try:
            print("🚀 Starting Advanced CS2 Trading Dashboard")
            print("=" * 50)
            
            # Initialize components
            if not self._initialize_components():
                print("❌ Failed to initialize components")
                return False
            
            # Start services
            self._start_services()
            
            # Enter main dashboard loop
            self._run_dashboard()
            
        except KeyboardInterrupt:
            print("\n👋 Shutting down dashboard...")
            self._cleanup()
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
            print(f"❌ Dashboard error: {e}")
            return False
    
    def _initialize_components(self) -> bool:
        """Initialize all trading system components"""
        try:
            print("🔧 Initializing components...")
            
            # Initialize authenticator
            self.authenticator = SteamAuthenticator()
            
            # Attempt Steam login
            if self.authenticator.login():
                self.is_authenticated = True
                print("✅ Steam authentication successful")
            else:
                print("⚠️ Steam authentication failed - some features will be limited")
                self.is_authenticated = False
            
            # Initialize components (some work without authentication)
            self.inventory_manager = InventoryManager(self.authenticator) if self.is_authenticated else None
            self.price_tracker = PriceTracker(self.authenticator)
            self.trade_evaluator = TradeOfferEvaluator(self.authenticator) if self.is_authenticated else None
            self.opportunity_detector = OpportunityDetector(self.authenticator) if self.is_authenticated else None
            self.float_analyzer = FloatAnalyzer()
            
            print("✅ Components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            print(f"❌ Component initialization failed: {e}")
            return False
    
    def _start_services(self):
        """Start background services"""
        try:
            print("🔄 Starting background services...")
            
            # Start price tracking
            self.price_tracker.add_alert_handler(telegram_alert_handler)
            self.price_tracker.start_tracking()
            print("✅ Price tracking started")
            
            # Start opportunity detection if authenticated
            if self.opportunity_detector:
                self.opportunity_detector.start_scanning()
                print("✅ Opportunity detection started")
            
            print("✅ All services started")
            
        except Exception as e:
            self.logger.error(f"Error starting services: {e}")
            print(f"⚠️ Some services failed to start: {e}")
    
    def _run_dashboard(self):
        """Main dashboard loop"""
        while True:
            try:
                # Update statistics
                self._update_stats()
                
                # Display current mode
                self._display_dashboard()
                
                # Handle user input
                if not self._handle_user_input():
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Dashboard loop error: {e}")
                print(f"❌ Error: {e}")
                time.sleep(2)
    
    def _update_stats(self):
        """Update dashboard statistics"""
        try:
            # Portfolio value
            if self.inventory_manager:
                portfolio = self.inventory_manager.get_portfolio_summary()
                self.stats.total_portfolio_value = portfolio.total_value
                self.stats.profit_potential = portfolio.potential_profit
            
            # Opportunities
            if self.opportunity_detector:
                opp_summary = self.opportunity_detector.get_opportunity_summary()
                self.stats.active_opportunities = opp_summary.get('total_opportunities', 0)
                self.stats.high_priority_opportunities = opp_summary.get('high_priority_count', 0)
            
            # Price tracking
            tracker_summary = self.price_tracker.get_tracking_summary()
            self.stats.active_price_alerts = tracker_summary.get('active_alerts_count', 0)
            self.stats.items_tracked = tracker_summary.get('watched_items_count', 0)
            self.stats.last_scan_time = tracker_summary.get('last_update')
            
            # System uptime
            uptime = datetime.now() - self.start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)
            self.stats.system_uptime = f"{hours}h {minutes}m"
            
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")
    
    def _display_dashboard(self):
        """Display the dashboard interface"""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Header
        print("🎯 CS2 Advanced Trading Dashboard")
        print("=" * 50)
        print(f"Mode: {self.current_mode.value.title()} | Uptime: {self.stats.system_uptime}")
        print(f"Status: {'🟢 Authenticated' if self.is_authenticated else '🔴 Not Authenticated'}")
        print("=" * 50)
        
        # Display mode-specific content
        if self.current_mode == DashboardMode.OVERVIEW:
            self._display_overview()
        elif self.current_mode == DashboardMode.PORTFOLIO:
            self._display_portfolio()
        elif self.current_mode == DashboardMode.OPPORTUNITIES:
            self._display_opportunities()
        elif self.current_mode == DashboardMode.PRICE_TRACKING:
            self._display_price_tracking()
        elif self.current_mode == DashboardMode.TRADE_EVALUATION:
            self._display_trade_evaluation()
        elif self.current_mode == DashboardMode.SETTINGS:
            self._display_settings()
        
        # Footer with navigation
        print("\n" + "=" * 50)
        print("Navigation: [1] Overview [2] Portfolio [3] Opportunities")
        print("           [4] Price Tracking [5] Trade Eval [6] Settings [Q] Quit")
    
    def _display_overview(self):
        """Display overview dashboard"""
        print("📊 SYSTEM OVERVIEW")
        print("-" * 30)
        
        # Key metrics
        print(f"💰 Portfolio Value: ${self.stats.total_portfolio_value:.2f}")
        print(f"📈 Profit Potential: ${self.stats.profit_potential:.2f}")
        print(f"🎯 Active Opportunities: {self.stats.active_opportunities}")
        print(f"⚠️ High Priority Opps: {self.stats.high_priority_opportunities}")
        print(f"🔔 Price Alerts: {self.stats.active_price_alerts}")
        print(f"👀 Items Tracked: {self.stats.items_tracked}")
        
        # Recent activity
        print("\n📋 RECENT ACTIVITY")
        print("-" * 30)
        
        if self.opportunity_detector:
            recent_opps = self.opportunity_detector.get_active_opportunities()[:5]
            if recent_opps:
                for i, opp in enumerate(recent_opps, 1):
                    print(f"{i}. {opp.opportunity_type.value}: {opp.item_name[:30]}")
                    print(f"   Profit: ${opp.profit_potential:.2f} ({opp.profit_percentage:.1%})")
            else:
                print("No recent opportunities")
        
        # System status
        print("\n🔧 SYSTEM STATUS")
        print("-" * 30)
        print(f"Price Tracking: {'🟢 Active' if self.price_tracker.is_tracking else '🔴 Stopped'}")
        if self.opportunity_detector:
            print(f"Opportunity Scanner: {'🟢 Active' if self.opportunity_detector.is_scanning else '🔴 Stopped'}")
        print(f"Last Scan: {self.stats.last_scan_time.strftime('%H:%M:%S') if self.stats.last_scan_time else 'Never'}")
    
    def _display_portfolio(self):
        """Display portfolio dashboard"""
        print("💼 PORTFOLIO MANAGEMENT")
        print("-" * 30)
        
        if not self.inventory_manager:
            print("❌ Portfolio not available (authentication required)")
            return
        
        try:
            # Get portfolio summary
            portfolio = self.inventory_manager.get_portfolio_summary()
            
            print(f"Total Items: {portfolio.total_items}")
            print(f"Total Value: ${portfolio.total_value:.2f}")
            print(f"Market Value: ${portfolio.total_market_value:.2f}")
            print(f"Potential Profit: ${portfolio.potential_profit:.2f}")
            
            # High value items
            print("\n💎 HIGH VALUE ITEMS")
            print("-" * 30)
            for i, item in enumerate(portfolio.high_value_items[:5], 1):
                print(f"{i}. {item.market_hash_name[:40]}")
                print(f"   Value: ${item.estimated_price:.2f} | Float: {item.float_value:.4f}")
            
            # Trading candidates
            print("\n📈 TRADING CANDIDATES")
            print("-" * 30)
            for i, item in enumerate(portfolio.trading_candidates[:5], 1):
                profit_pct = (item.profit_potential / max(item.market_price, 1)) * 100
                print(f"{i}. {item.market_hash_name[:40]}")
                print(f"   Profit: ${item.profit_potential:.2f} ({profit_pct:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error loading portfolio: {e}")
    
    def _display_opportunities(self):
        """Display opportunities dashboard"""
        print("🎯 MARKET OPPORTUNITIES")
        print("-" * 30)
        
        if not self.opportunity_detector:
            print("❌ Opportunities not available (authentication required)")
            return
        
        try:
            # Get opportunities by priority
            critical_opps = self.opportunity_detector.get_active_opportunities(priority=OpportunityPriority.CRITICAL)
            high_opps = self.opportunity_detector.get_active_opportunities(priority=OpportunityPriority.HIGH)
            medium_opps = self.opportunity_detector.get_active_opportunities(priority=OpportunityPriority.MEDIUM)
            
            # Display critical opportunities
            if critical_opps:
                print("🚨 CRITICAL OPPORTUNITIES")
                for i, opp in enumerate(critical_opps[:3], 1):
                    print(f"{i}. {opp.item_name[:35]}")
                    print(f"   Type: {opp.opportunity_type.value}")
                    print(f"   Profit: ${opp.profit_potential:.2f} ({opp.profit_percentage:.1%})")
                    print(f"   Action: {opp.action_required[:50]}")
                    print()
            
            # Display high priority opportunities
            if high_opps:
                print("⚠️ HIGH PRIORITY OPPORTUNITIES")
                for i, opp in enumerate(high_opps[:3], 1):
                    print(f"{i}. {opp.item_name[:35]}")
                    print(f"   Profit: ${opp.profit_potential:.2f} | Risk: {opp.risk_level.value}")
                    print()
            
            # Summary stats
            print(f"Total Opportunities: {len(critical_opps + high_opps + medium_opps)}")
            print(f"Total Profit Potential: ${sum(opp.profit_potential for opp in critical_opps + high_opps + medium_opps):.2f}")
            
        except Exception as e:
            print(f"❌ Error loading opportunities: {e}")
    
    def _display_price_tracking(self):
        """Display price tracking dashboard"""
        print("📊 PRICE TRACKING")
        print("-" * 30)
        
        try:
            # Get tracking summary
            summary = self.price_tracker.get_tracking_summary()
            
            print(f"Tracking Status: {'🟢 Active' if summary.get('is_tracking') else '🔴 Stopped'}")
            print(f"Watched Items: {summary.get('watched_items_count', 0)}")
            print(f"Active Alerts: {summary.get('active_alerts_count', 0)}")
            print(f"Items with Prices: {summary.get('items_with_prices', 0)}")
            
            # Recent price updates
            print("\n💰 RECENT PRICES")
            print("-" * 30)
            
            # Show some current prices
            sample_items = list(self.price_tracker.watched_items)[:5]
            for item in sample_items:
                current_price = self.price_tracker.get_current_price(item)
                if current_price > 0:
                    print(f"{item[:35]}: ${current_price:.2f}")
            
            # Active alerts
            active_alerts = [alert for alert in self.price_tracker.price_alerts.values() if alert.is_active]
            if active_alerts:
                print("\n🔔 ACTIVE ALERTS")
                print("-" * 30)
                for alert in active_alerts[:3]:
                    print(f"{alert.alert_type.value}: {alert.item_name}")
                    print(f"Threshold: ${alert.threshold_price:.2f}")
            
        except Exception as e:
            print(f"❌ Error loading price tracking: {e}")
    
    def _display_trade_evaluation(self):
        """Display trade evaluation dashboard"""
        print("⚖️ TRADE EVALUATION")
        print("-" * 30)
        
        if not self.trade_evaluator:
            print("❌ Trade evaluation not available (authentication required)")
            return
        
        try:
            # Show recent evaluations
            recent_evaluations = self.trade_evaluator.evaluation_history[-5:]
            
            if recent_evaluations:
                print("📋 RECENT EVALUATIONS")
                for i, eval in enumerate(recent_evaluations, 1):
                    print(f"{i}. Offer {eval.offer_id}")
                    print(f"   Recommendation: {eval.recommendation.value.upper()}")
                    print(f"   Net Value: ${eval.net_value:.2f} ({eval.profit_percentage:.1%})")
                    print(f"   Risk: {eval.risk_level.value} | Confidence: {eval.confidence_score:.1%}")
                    print()
            else:
                print("No recent trade evaluations")
            
            print("\n🎯 EVALUATION GUIDELINES")
            print("-" * 30)
            print(f"Min Profit Threshold: {self.trade_evaluator.min_profit_threshold:.1%}")
            print(f"Auto-Accept Threshold: {self.trade_evaluator.auto_accept_threshold:.1%}")
            print(f"Max Risk for Auto: {self.trade_evaluator.max_risk_for_auto.value}")
            
        except Exception as e:
            print(f"❌ Error loading trade evaluation: {e}")
    
    def _display_settings(self):
        """Display settings dashboard"""
        print("⚙️ SYSTEM SETTINGS")
        print("-" * 30)
        
        # Authentication status
        print("🔐 AUTHENTICATION")
        print(f"Steam Account: {'✅ Connected' if self.is_authenticated else '❌ Not Connected'}")
        if self.is_authenticated:
            steam_id = self.authenticator.get_steam_id()
            print(f"Steam ID: {steam_id}")
        
        # Component status
        print("\n🔧 COMPONENTS")
        print(f"Inventory Manager: {'✅' if self.inventory_manager else '❌'}")
        print(f"Price Tracker: {'✅' if self.price_tracker else '❌'}")
        print(f"Trade Evaluator: {'✅' if self.trade_evaluator else '❌'}")
        print(f"Opportunity Detector: {'✅' if self.opportunity_detector else '❌'}")
        
        # Service status
        print("\n🔄 SERVICES")
        print(f"Price Tracking: {'🟢 Running' if self.price_tracker.is_tracking else '🔴 Stopped'}")
        if self.opportunity_detector:
            print(f"Opportunity Scanner: {'🟢 Running' if self.opportunity_detector.is_scanning else '🔴 Stopped'}")
        
        # Configuration
        print("\n📋 CONFIGURATION")
        print(f"Steam API Key: {'✅ Set' if self.config.STEAM_API_KEY else '❌ Missing'}")
        print(f"Telegram Bot: {'✅ Set' if self.config.TELEGRAM_BOT_TOKEN else '❌ Missing'}")
        print(f"Update Interval: {self.price_tracker.update_interval}s")
    
    def _handle_user_input(self):
        """Handle user input and navigation"""
        try:
            print("\nEnter command: ", end="", flush=True)
            choice = input().strip().lower()
            
            if choice == 'q' or choice == 'quit':
                return False
            elif choice == '1':
                self.current_mode = DashboardMode.OVERVIEW
            elif choice == '2':
                self.current_mode = DashboardMode.PORTFOLIO
            elif choice == '3':
                self.current_mode = DashboardMode.OPPORTUNITIES
            elif choice == '4':
                self.current_mode = DashboardMode.PRICE_TRACKING
            elif choice == '5':
                self.current_mode = DashboardMode.TRADE_EVALUATION
            elif choice == '6':
                self.current_mode = DashboardMode.SETTINGS
            elif choice == 'refresh' or choice == 'r':
                self._update_stats()
            elif choice == 'export':
                self._export_data()
            elif choice.startswith('alert '):
                self._create_alert_from_command(choice)
            else:
                print(f"Unknown command: {choice}")
                time.sleep(1)
            
            return True
            
        except KeyboardInterrupt:
            return False
        except Exception as e:
            self.logger.error(f"Error handling user input: {e}")
            print(f"Error: {e}")
            time.sleep(1)
            return True
    
    def _create_alert_from_command(self, command: str):
        """Create price alert from command"""
        try:
            # Parse command: alert <item_name> <type> <threshold>
            parts = command.split(' ', 3)
            if len(parts) < 4:
                print("Usage: alert <item_name> <drop|spike> <threshold>")
                return
            
            item_name = parts[1].replace('_', ' ')
            alert_type_str = parts[2]
            threshold = float(parts[3])
            
            alert_type = AlertType.PRICE_DROP if alert_type_str == 'drop' else AlertType.PRICE_SPIKE
            
            alert_id = self.price_tracker.create_price_alert(item_name, alert_type, threshold)
            if alert_id:
                print(f"✅ Alert created: {alert_id}")
            else:
                print("❌ Failed to create alert")
                
        except Exception as e:
            print(f"❌ Error creating alert: {e}")
    
    def _export_data(self):
        """Export current data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export opportunities
            if self.opportunity_detector:
                opp_file = self.opportunity_detector.export_opportunities(f"opportunities_{timestamp}.json")
                if opp_file:
                    print(f"✅ Opportunities exported to {opp_file}")
            
            # Export portfolio
            if self.inventory_manager:
                portfolio_file = self.inventory_manager.export_portfolio_report(f"portfolio_{timestamp}.json")
                if portfolio_file:
                    print(f"✅ Portfolio exported to {portfolio_file}")
            
            # Export price data
            price_file = self.price_tracker.export_price_data(filename=f"prices_{timestamp}.csv")
            if price_file:
                print(f"✅ Price data exported to {price_file}")
                
        except Exception as e:
            print(f"❌ Export error: {e}")
    
    def _cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            print("🔄 Cleaning up...")
            
            # Stop services
            if self.price_tracker:
                self.price_tracker.stop_tracking()
            
            if self.opportunity_detector:
                self.opportunity_detector.stop_scanning()
            
            # Logout from Steam
            if self.authenticator:
                self.authenticator.logout()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            print(f"⚠️ Cleanup error: {e}")

def main():
    """Main entry point for the trading dashboard"""
    try:
        dashboard = TradingDashboard()
        dashboard.start()
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())