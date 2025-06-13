#!/usr/bin/env python3

import asyncio
import logging
import time
import json
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, asdict
import sys
import argparse

from optimized_steam_api import OptimizedSteamAPI
from float_analyzer import FloatAnalyzer, FloatAnalysis
from database import FloatDatabase
from config import FloatCheckerConfig
from skin_database import SkinDatabase
from telegram_bot import TelegramNotifier

@dataclass
class ScanningStats:
    session_start: datetime
    items_scanned: int = 0
    rare_items_found: int = 0
    total_value_found: float = 0.0
    requests_made: int = 0
    errors_encountered: int = 0
    current_scan_rate: float = 0.0
    best_find: Optional[FloatAnalysis] = None
    weapons_covered: Set[str] = None
    
    def __post_init__(self):
        if self.weapons_covered is None:
            self.weapons_covered = set()

class EnhancedFloatChecker:
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.analyzer = FloatAnalyzer()
        self.database = FloatDatabase()
        self.skin_db = SkinDatabase()
        self.telegram = TelegramNotifier()
        self.setup_logging()
        
        # Enhanced statistics
        self.stats = ScanningStats(session_start=datetime.now())
        self.performance_lock = threading.Lock()
        
        # Scanning state
        self.should_stop = False
        self.current_priority_items = []
        self.scanned_recently = set()  # Items scanned in last hour
        
        # Performance tracking
        self.last_stats_update = time.time()
        self.scan_queue = asyncio.Queue()
    
    def setup_logging(self):
        """Enhanced logging setup"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Set specific log levels for optimization
        if self.config.AGGRESSIVE_SCANNING_MODE:
            logging.getLogger('aiohttp').setLevel(logging.WARNING)
            logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    async def scan_entire_market_optimized(self) -> Dict[str, List[FloatAnalysis]]:
        """Scan the entire CS2 market with maximum efficiency"""
        self.logger.info("🚀 Starting optimized full market scan...")
        self.telegram.send_startup_notification()
        
        results = {}
        scan_start_time = time.time()
        
        async with OptimizedSteamAPI() as steam_api:
            # Get all available items from skin database
            all_items = self._get_prioritized_scan_list()
            total_items = len(all_items)
            
            self.logger.info(f"📊 Scanning {total_items} items from complete CS2 database")
            
            # Process items in optimized batches
            batch_size = self.config.MAX_CONCURRENT_REQUESTS * 5
            
            for i in range(0, total_items, batch_size):
                if self.should_stop:
                    break
                
                batch = all_items[i:i + batch_size]
                batch_results = await self._process_item_batch_optimized(steam_api, batch)
                
                for item_name, analyses in batch_results.items():
                    if analyses:
                        results[item_name] = analyses
                
                # Update progress
                progress = ((i + batch_size) / total_items) * 100
                self._update_scanning_stats(steam_api, progress)
                
                # Brief pause between large batches to prevent overwhelming
                if i % (batch_size * 3) == 0:
                    await asyncio.sleep(0.5)
        
        scan_duration = time.time() - scan_start_time
        self._log_final_results(scan_duration, results)
        
        return results
    
    def _get_prioritized_scan_list(self) -> List[str]:
        """Get prioritized list of all CS2 items to scan"""
        all_items = []
        
        # Get all skins from database
        db_stats = self.skin_db.get_database_stats()
        self.logger.info(f"📋 Database contains {db_stats['total_skins']} skins across {db_stats['total_weapons']} weapons")
        
        # Priority 1: High-value items (knives, gloves, expensive skins)
        priority_weapons = [
            "★", "Karambit", "M9 Bayonet", "Butterfly Knife", "Bayonet", "Flip Knife",
            "AWP", "AK-47", "M4A4", "M4A1-S", "Specialist Gloves", "Sport Gloves"
        ]
        
        priority_items = []
        regular_items = []
        
        # Categorize items by priority
        for weapon in self.skin_db.get_all_weapons():
            weapon_skins = self.skin_db.get_skins_by_weapon(weapon)
            
            is_priority = any(priority_weapon in weapon for priority_weapon in priority_weapons)
            
            for skin in weapon_skins:
                if is_priority:
                    priority_items.append(skin.name)
                else:
                    regular_items.append(skin.name)
        
        # Return prioritized list
        all_items = priority_items + regular_items
        
        # Remove recently scanned items from the front of the queue
        current_time = time.time()
        recently_scanned = {item for item, scan_time in self.scanned_recently 
                          if current_time - scan_time < 3600}  # 1 hour
        
        # Move recently scanned items to the end
        not_recent = [item for item in all_items if item not in recently_scanned]
        recent = [item for item in all_items if item in recently_scanned]
        
        self.logger.info(f"🎯 Priority items: {len(priority_items)}, Regular items: {len(regular_items)}")
        self.logger.info(f"⏰ Recently scanned (moving to end): {len(recent)}")
        
        return not_recent + recent
    
    async def _process_item_batch_optimized(self, steam_api: OptimizedSteamAPI, item_batch: List[str]) -> Dict[str, List[FloatAnalysis]]:
        """Process a batch of items with maximum efficiency"""
        results = {}
        
        # Process listings in parallel
        async for item_name, listings_data in steam_api.get_item_listings_batch(
            item_batch, 
            count=min(100, self.config.MAX_CONCURRENT_REQUESTS * 20)
        ):
            try:
                analyses = await self._analyze_item_listings_fast(item_name, listings_data)
                if analyses:
                    results[item_name] = analyses
                    
                    # Track weapon coverage
                    weapon = item_name.split(' | ')[0] if ' | ' in item_name else item_name
                    self.stats.weapons_covered.add(weapon)
                
                # Mark as recently scanned
                self.scanned_recently.add((item_name, time.time()))
                
            except Exception as e:
                self.logger.error(f"Error processing {item_name}: {e}")
                self.stats.errors_encountered += 1
        
        return results
    
    async def _analyze_item_listings_fast(self, item_name: str, listings_data: Dict) -> List[FloatAnalysis]:
        """Fast analysis of item listings"""
        analyses = []
        
        if not listings_data:
            return analyses
        
        # Extract inspect links
        inspect_links = OptimizedSteamAPI().extract_inspect_links(listings_data)
        
        # For demo purposes, simulate float extraction
        # In production, you'd integrate with CSFloat API or similar
        for i, inspect_link in enumerate(inspect_links[:50]):  # Limit per item
            simulated_float = self._simulate_float_value_smart(item_name)
            simulated_price = 10.0 + (i * 2.5)  # Simulate price progression
            
            if simulated_float is not None:
                analysis = self.analyzer.analyze_float_rarity(
                    item_name, simulated_float, simulated_price
                )
                analysis.inspect_link = inspect_link
                
                analyses.append(analysis)
                
                # Save to database
                self.database.save_analysis(analysis)
                
                self.stats.items_scanned += 1
                
                if analysis.is_rare:
                    self.stats.rare_items_found += 1
                    self.stats.total_value_found += analysis.price
                    
                    # Update best find
                    if not self.stats.best_find or analysis.rarity_score > self.stats.best_find.rarity_score:
                        self.stats.best_find = analysis
                    
                    # Send immediate notification for very rare items
                    if analysis.rarity_score >= 90:
                        await self._send_urgent_notification(analysis)
        
        return analyses
    
    def _simulate_float_value_smart(self, item_name: str) -> Optional[float]:
        """Smarter float value simulation based on actual skin ranges"""
        import random
        
        # Get skin info for accurate ranges
        skin_info = self.skin_db.get_skin_info(item_name)
        
        if skin_info and skin_info.wear_ranges:
            # Choose a random wear condition from available ones
            available_wears = list(skin_info.wear_ranges.keys())
            if not available_wears:
                return None
            
            wear_condition = random.choice(available_wears)
            min_val, max_val = skin_info.wear_ranges[wear_condition]
            
            # 15% chance of generating a rare float
            if random.random() < 0.15:
                if wear_condition == "Factory New":
                    # Generate very low float
                    return random.uniform(min_val, min(min_val + 0.01, max_val))
                elif wear_condition == "Battle-Scarred":
                    # Generate very high float
                    return random.uniform(max(max_val - 0.01, min_val), max_val)
            
            # Normal float within range
            return random.uniform(min_val, max_val)
        
        # Fallback to standard ranges
        wear_ranges = self.config.WEAR_RANGES
        wear_condition = random.choice(list(wear_ranges.keys()))
        min_val, max_val = wear_ranges[wear_condition]
        
        return random.uniform(min_val, max_val)
    
    async def _send_urgent_notification(self, analysis: FloatAnalysis):
        """Send urgent notification for extremely rare finds"""
        if analysis.rarity_score >= 95:
            # Immediate notification for legendary finds
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.telegram.send_rare_float_alert, 
                analysis
            )
    
    def _update_scanning_stats(self, steam_api: OptimizedSteamAPI, progress: float):
        """Update and display scanning statistics"""
        current_time = time.time()
        
        with self.performance_lock:
            # Update scan rate
            time_elapsed = current_time - self.last_stats_update
            if time_elapsed >= 10.0:  # Update every 10 seconds
                
                # Get API performance stats
                api_stats = steam_api.get_performance_stats()
                self.stats.requests_made = api_stats['total_requests']
                self.stats.current_scan_rate = api_stats['requests_per_second']
                
                # Log progress
                self.logger.info(
                    f"📈 Progress: {progress:.1f}% | "
                    f"Items: {self.stats.items_scanned} | "
                    f"Rare: {self.stats.rare_items_found} | "
                    f"Rate: {self.stats.current_scan_rate:.1f} req/s | "
                    f"Success: {api_stats['success_rate']:.1f}% | "
                    f"Weapons: {len(self.stats.weapons_covered)}"
                )
                
                self.last_stats_update = current_time
    
    def _log_final_results(self, scan_duration: float, results: Dict):
        """Log comprehensive final results"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🏁 ENHANCED MARKET SCAN COMPLETED")
        self.logger.info("="*60)
        
        total_analyses = sum(len(analyses) for analyses in results.values())
        
        self.logger.info(f"⏱️ Scan Duration: {timedelta(seconds=int(scan_duration))}")
        self.logger.info(f"📊 Items Processed: {len(results)}")
        self.logger.info(f"🔍 Total Analyses: {total_analyses}")
        self.logger.info(f"⭐ Rare Items Found: {self.stats.rare_items_found}")
        self.logger.info(f"💰 Total Value Found: ${self.stats.total_value_found:.2f}")
        self.logger.info(f"🎯 Weapons Covered: {len(self.stats.weapons_covered)}")
        self.logger.info(f"📡 Requests Made: {self.stats.requests_made}")
        self.logger.info(f"⚡ Average Rate: {self.stats.requests_made / scan_duration:.1f} req/s")
        
        if self.stats.best_find:
            self.logger.info(f"🏆 Best Find: {self.stats.best_find.item_name} "
                           f"(Score: {self.stats.best_find.rarity_score:.1f})")
        
        # Send summary to Telegram
        summary_stats = {
            'items_scanned': total_analyses,
            'rare_items_found': self.stats.rare_items_found,
            'total_value': self.stats.total_value_found,
            'scan_duration': str(timedelta(seconds=int(scan_duration))),
            'errors': self.stats.errors_encountered,
            'best_find': f"{self.stats.best_find.item_name} (Score: {self.stats.best_find.rarity_score:.1f})" if self.stats.best_find else "None",
            'weapon_stats': {weapon: 1 for weapon in list(self.stats.weapons_covered)[:10]}  # Top 10 for display
        }
        
        self.telegram.send_daily_summary(summary_stats)
    
    async def continuous_aggressive_scan(self, interval_minutes: int = 5):
        """Continuous scanning with minimal delays"""
        self.logger.info(f"🔄 Starting continuous aggressive scanning (interval: {interval_minutes} min)")
        
        while not self.should_stop:
            try:
                # Full market scan
                await self.scan_entire_market_optimized()
                
                # Brief pause before next cycle
                self.logger.info(f"💤 Waiting {interval_minutes} minutes before next scan cycle...")
                await asyncio.sleep(interval_minutes * 60)
                
                # Clean up old "recently scanned" items
                current_time = time.time()
                self.scanned_recently = {
                    (item, scan_time) for item, scan_time in self.scanned_recently
                    if current_time - scan_time < 3600  # Keep for 1 hour
                }
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Stopping continuous scan...")
                self.should_stop = True
                break
            except Exception as e:
                self.logger.error(f"❌ Error in continuous scan: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def stop_scanning(self):
        """Stop all scanning operations"""
        self.should_stop = True
        self.logger.info("🛑 Stop signal sent to all scanning operations")

async def main():
    parser = argparse.ArgumentParser(description='Enhanced CS2 Float Checker - Maximum Speed Market Scanner')
    parser.add_argument('--full-scan', action='store_true', help='Scan entire CS2 market once')
    parser.add_argument('--continuous', action='store_true', help='Continuous aggressive scanning')
    parser.add_argument('--interval', type=int, default=5, help='Continuous scan interval in minutes (default: 5)')
    parser.add_argument('--test-performance', action='store_true', help='Test API performance')
    
    args = parser.parse_args()
    
    checker = EnhancedFloatChecker()
    
    try:
        if args.test_performance:
            # Test performance with a small batch
            async with OptimizedSteamAPI() as api:
                start_time = time.time()
                async for term, result in api.search_items_batch(["AK-47", "AWP", "M4A4"], count=10):
                    print(f"✅ {term}: {len(result.get('results', [])) if result else 0} results")
                
                duration = time.time() - start_time
                stats = api.get_performance_stats()
                print(f"\n⚡ Performance Test Results:")
                print(f"Duration: {duration:.2f}s")
                print(f"Requests: {stats['total_requests']}")
                print(f"Success Rate: {stats['success_rate']:.1f}%")
                print(f"Avg Response Time: {stats['average_response_time']:.3f}s")
        
        elif args.full_scan:
            await checker.scan_entire_market_optimized()
        
        elif args.continuous:
            await checker.continuous_aggressive_scan(args.interval)
        
        else:
            print("Use --full-scan for one-time scan or --continuous for ongoing monitoring")
    
    except KeyboardInterrupt:
        checker.stop_scanning()
        print("\n👋 Scanning stopped by user")

if __name__ == "__main__":
    asyncio.run(main())