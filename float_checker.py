#!/usr/bin/env python3

import asyncio
import logging
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import argparse
import sys

from steam_market_api import SteamMarketAPI
from float_analyzer import FloatAnalyzer, FloatAnalysis
from database import FloatDatabase
from config import FloatCheckerConfig
from skin_database import SkinDatabase
from telegram_bot import TelegramNotifier

class CS2FloatChecker:
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.steam_api = SteamMarketAPI()
        self.analyzer = FloatAnalyzer()
        self.database = FloatDatabase()
        self.skin_db = SkinDatabase()
        self.telegram = TelegramNotifier()
        self.setup_logging()
        
        # Statistics
        self.stats = {
            'items_checked': 0,
            'rare_items_found': 0,
            'total_value_found': 0.0,
            'start_time': None,
            'errors': 0,
            'weapon_stats': {},
            'best_find': None
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def scan_item(self, item_name: str, max_listings: int = 100) -> List[FloatAnalysis]:
        """Scan a specific item for rare floats"""
        self.logger.info(f"Scanning {item_name}...")
        analyses = []
        
        try:
            # Get all wear variants for this item
            variants = self.steam_api.get_all_skin_variants(item_name)
            
            for variant in variants:
                self.logger.info(f"Checking variant: {variant}")
                
                # Get market listings
                listings = self.steam_api.get_item_listings(variant, count=max_listings)
                
                if not listings:
                    self.logger.warning(f"No listings found for {variant}")
                    continue
                
                # Extract inspect links
                inspect_links = self.steam_api.extract_inspect_links(listings)
                self.logger.info(f"Found {len(inspect_links)} inspect links for {variant}")
                
                # Process each inspect link (placeholder for float extraction)
                for i, inspect_link in enumerate(inspect_links[:10]):  # Limit to first 10 for demo
                    # In a real implementation, you'd call a third-party API here
                    # For now, simulate float values
                    simulated_float = self.simulate_float_value(variant)
                    simulated_price = 10.0 + (i * 5.0)  # Simulate price
                    
                    if simulated_float is not None:
                        analysis = self.analyzer.analyze_float_rarity(
                            variant, simulated_float, simulated_price
                        )
                        analysis.inspect_link = inspect_link
                        analyses.append(analysis)
                        
                        # Save to database
                        self.database.save_analysis(analysis)
                        
                        self.stats['items_checked'] += 1
                        
                        # Update weapon stats
                        weapon = analysis.item_name.split(' | ')[0] if ' | ' in analysis.item_name else analysis.item_name
                        self.stats['weapon_stats'][weapon] = self.stats['weapon_stats'].get(weapon, 0) + 1
                        
                        if analysis.is_rare:
                            self.stats['rare_items_found'] += 1
                            self.stats['total_value_found'] += analysis.price
                            
                            # Update best find
                            if not self.stats['best_find'] or analysis.rarity_score > self.stats['best_find'].rarity_score:
                                self.stats['best_find'] = analysis
                            
                            self.logger.info(f"RARE ITEM FOUND: {analysis.item_name} - Float: {analysis.float_value:.6f} - Score: {analysis.rarity_score}")
                            
                            # Send Telegram notification
                            self.telegram.send_rare_float_alert(analysis)
                
                # Rate limiting
                self.steam_api.rate_limit_delay()
        
        except Exception as e:
            self.logger.error(f"Error scanning {item_name}: {e}")
            self.stats['errors'] += 1
        
        return analyses
    
    def simulate_float_value(self, item_name: str) -> Optional[float]:
        """Simulate float value extraction (placeholder)"""
        # In a real implementation, this would call a third-party API
        # For demo purposes, generate random float values within wear ranges
        import random
        
        # Extract wear condition from item name
        wear_condition = "Factory New"  # Default
        for wear in self.config.WEAR_RANGES.keys():
            if wear in item_name:
                wear_condition = wear
                break
        
        # Get range for this wear condition
        if wear_condition in self.config.WEAR_RANGES:
            min_val, max_val = self.config.WEAR_RANGES[wear_condition]
            
            # Occasionally generate rare floats
            if random.random() < 0.05:  # 5% chance of rare float
                if wear_condition == "Factory New":
                    return random.uniform(0.0, 0.005)  # Very low float
                elif wear_condition == "Battle-Scarred":
                    return random.uniform(0.995, 1.0)  # Very high float
            
            # Normal float within range
            return random.uniform(min_val, max_val)
        
        return None
    
    def scan_multiple_items(self, item_names: List[str], max_listings_per_item: int = 50) -> Dict[str, List[FloatAnalysis]]:
        """Scan multiple items"""
        self.stats['start_time'] = datetime.now()
        results = {}
        
        self.logger.info(f"Starting scan of {len(item_names)} items...")
        self.telegram.send_startup_notification()
        
        for i, item_name in enumerate(item_names, 1):
            self.logger.info(f"Progress: {i}/{len(item_names)} - {item_name}")
            
            analyses = self.scan_item(item_name, max_listings_per_item)
            results[item_name] = analyses
            
            # Print progress
            self.print_progress_stats()
        
        self.logger.info("Scan completed!")
        self.print_final_stats()
        
        # Send completion summary
        summary_stats = self._prepare_summary_stats()
        self.telegram.send_daily_summary(summary_stats)
        
        return results
    
    def continuous_scan(self, item_names: List[str], interval_minutes: int = 30):
        """Run continuous scanning"""
        self.logger.info(f"Starting continuous scan with {interval_minutes} minute intervals...")
        
        while True:
            try:
                self.scan_multiple_items(item_names)
                
                self.logger.info(f"Waiting {interval_minutes} minutes before next scan...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Continuous scan stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous scan: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def print_progress_stats(self):
        """Print current progress statistics"""
        if self.stats['start_time']:
            elapsed = datetime.now() - self.stats['start_time']
            elapsed_minutes = elapsed.total_seconds() / 60
            
            print(f"\n--- Progress Stats ---")
            print(f"Items checked: {self.stats['items_checked']}")
            print(f"Rare items found: {self.stats['rare_items_found']}")
            print(f"Total value of rare items: ${self.stats['total_value_found']:.2f}")
            print(f"Elapsed time: {elapsed_minutes:.1f} minutes")
            print(f"Items per minute: {self.stats['items_checked'] / elapsed_minutes:.1f}" if elapsed_minutes > 0 else "Items per minute: N/A")
            print(f"Errors: {self.stats['errors']}")
            print("-" * 20)
    
    def print_final_stats(self):
        """Print final statistics"""
        if self.stats['start_time']:
            elapsed = datetime.now() - self.stats['start_time']
            
            print(f"\n{'='*50}")
            print(f"FINAL SCAN RESULTS")
            print(f"{'='*50}")
            print(f"Total items checked: {self.stats['items_checked']}")
            print(f"Rare items found: {self.stats['rare_items_found']}")
            print(f"Rare item percentage: {(self.stats['rare_items_found'] / self.stats['items_checked'] * 100):.2f}%" if self.stats['items_checked'] > 0 else "Rare item percentage: N/A")
            print(f"Total value of rare items: ${self.stats['total_value_found']:.2f}")
            print(f"Total scan time: {elapsed}")
            print(f"Errors encountered: {self.stats['errors']}")
            print(f"{'='*50}")
    
    def get_recent_rare_finds(self, hours: int = 24) -> List[Dict]:
        """Get recent rare finds from database"""
        return self.database.get_rare_items(min_rarity_score=70.0, limit=100)
    
    def export_rare_finds(self, filename: str = "rare_finds.json"):
        """Export rare finds to JSON file"""
        import json
        
        rare_items = self.get_recent_rare_finds()
        
        with open(filename, 'w') as f:
            json.dump(rare_items, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(rare_items)} rare finds to {filename}")
    
    def _prepare_summary_stats(self) -> Dict:
        """Prepare summary statistics for notifications"""
        runtime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        
        return {
            'items_scanned': self.stats['items_checked'],
            'rare_items_found': self.stats['rare_items_found'],
            'total_value': self.stats['total_value_found'],
            'scan_duration': str(runtime).split('.')[0],  # Remove microseconds
            'errors': self.stats['errors'],
            'best_find': f"{self.stats['best_find'].item_name} (Score: {self.stats['best_find'].rarity_score:.1f})" if self.stats['best_find'] else "None",
            'weapon_stats': self.stats['weapon_stats']
        }
    
    def scan_all_weapons(self, max_items_per_weapon: int = 20) -> Dict[str, List[FloatAnalysis]]:
        """Scan all available weapons from the database"""
        self.logger.info("Starting comprehensive weapon scan...")
        
        all_weapons = self.skin_db.get_all_weapons()
        self.logger.info(f"Found {len(all_weapons)} weapon types in database")
        
        results = {}
        total_items = []
        
        # Get representative skins for each weapon
        for weapon in all_weapons[:50]:  # Limit to first 50 weapons to avoid overwhelming
            weapon_skins = self.skin_db.get_skins_by_weapon(weapon)
            
            # Select most popular/valuable skins for each weapon
            selected_skins = weapon_skins[:max_items_per_weapon]
            
            for skin in selected_skins:
                total_items.append(skin.name)
        
        self.logger.info(f"Selected {len(total_items)} items to scan")
        
        return self.scan_multiple_items(total_items)
    
    def test_telegram(self) -> bool:
        """Test Telegram connection"""
        return self.telegram.test_connection()

def main():
    parser = argparse.ArgumentParser(description='CS2 Float Checker - Scan Steam Market for rare float values')
    parser.add_argument('--items', nargs='+', help='Specific items to scan')
    parser.add_argument('--continuous', action='store_true', help='Run continuous scanning')
    parser.add_argument('--interval', type=int, default=30, help='Interval between scans in minutes (default: 30)')
    parser.add_argument('--max-listings', type=int, default=50, help='Max listings to check per item (default: 50)')
    parser.add_argument('--export', type=str, help='Export rare finds to JSON file')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--all-weapons', action='store_true', help='Scan all weapons from database')
    parser.add_argument('--test-telegram', action='store_true', help='Test Telegram notifications')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced optimized scanning')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive scanning mode (fastest)')
    parser.add_argument('--full-market', action='store_true', help='Scan entire CS2 market (enhanced mode only)')
    
    args = parser.parse_args()
    
    checker = CS2FloatChecker()
    
    if args.stats:
        stats = checker.database.get_statistics()
        print("\n--- Database Statistics ---")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        return
    
    if args.export:
        checker.export_rare_finds(args.export)
        return
    
    if args.test_telegram:
        success = checker.test_telegram()
        print("✅ Telegram test successful!" if success else "❌ Telegram test failed!")
        return
    
    if args.enhanced or args.aggressive or args.full_market:
        # Use enhanced scanner
        import asyncio
        from enhanced_float_checker import EnhancedFloatChecker
        enhanced_checker = EnhancedFloatChecker()
        
        if args.full_market:
            asyncio.run(enhanced_checker.scan_entire_market_optimized())
        elif args.continuous:
            interval = 5 if args.aggressive else args.interval
            asyncio.run(enhanced_checker.continuous_aggressive_scan(interval))
        else:
            asyncio.run(enhanced_checker.scan_entire_market_optimized())
        return
    
    if args.all_weapons:
        checker.scan_all_weapons()
        return
    
    # Determine items to scan
    items_to_scan = args.items if args.items else checker.config.MONITORED_ITEMS
    
    if args.continuous:
        checker.continuous_scan(items_to_scan, args.interval)
    else:
        checker.scan_multiple_items(items_to_scan, args.max_listings)

if __name__ == "__main__":
    main()