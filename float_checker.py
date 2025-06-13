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
from csfloat_api import CSFloatAPI, FloatData, get_float_data

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
    
    def scan_item_extreme_floats(self, item_name: str, max_listings: int = 100) -> List[FloatAnalysis]:
        """Scan a specific item for ONLY extreme float values (FN < 0.0001 or BS > 0.99)"""
        self.logger.info(f"🎯 Scanning {item_name} for extreme floats only...")
        analyses = []
        
        try:
            # Get only Factory New and Battle-Scarred variants that exist for this item
            extreme_variants = self.steam_api.get_extreme_float_variants(item_name)
            
            if not extreme_variants:
                self.logger.warning(f"No valid extreme float variants found for {item_name}")
                return analyses
            
            self.logger.info(f"Extreme variants to scan: {extreme_variants}")
            
            for variant in extreme_variants:
                self.logger.info(f"🔍 Checking variant: {variant}")
                
                # Get market listings
                listings = self.steam_api.get_item_listings(variant, count=max_listings)
                
                if not listings:
                    self.logger.warning(f"No listings found for {variant}")
                    continue
                
                # Extract inspect links
                inspect_links = self.steam_api.extract_inspect_links(listings)
                self.logger.info(f"Found {len(inspect_links)} inspect links for {variant}")
                
                # Process each inspect link with REAL CSFloat API
                extreme_found = 0
                self.logger.info(f"🔍 Processing {len(inspect_links)} inspect links with CSFloat API...")
                
                # Use async processing for real float extraction
                real_floats = asyncio.run(self._get_real_floats_batch(inspect_links[:15]))
                
                for i, (inspect_link, float_data) in enumerate(zip(inspect_links[:15], real_floats)):
                    if float_data is None:
                        self.logger.debug(f"⚠️ No float data for inspect link {i+1}")
                        continue
                    
                    # Use REAL float value from CSFloat API
                    real_float = float_data.float_value
                    estimated_price = self._estimate_price_from_float_data(float_data)
                    
                    self.logger.debug(f"📊 Real float: {real_float:.6f} for {variant}")
                    
                    # Only process if it's actually an extreme float
                    if self.steam_api.is_extreme_float_candidate(real_float, variant):
                        analysis = self.analyzer.analyze_float_rarity(
                            variant, real_float, estimated_price
                        )
                        analysis.inspect_link = inspect_link
                        
                        # Add CSFloat-specific data
                        analysis.paint_seed = float_data.paint_seed
                        analysis.paint_index = float_data.paint_index
                        analysis.rank_info = f"Rank {float_data.low_rank}-{float_data.high_rank}"
                        
                        analyses.append(analysis)
                        
                        # Save to database
                        self.database.save_analysis(analysis)
                        
                        self.stats['items_checked'] += 1
                        extreme_found += 1
                        
                        # Update weapon stats
                        weapon = analysis.item_name.split(' | ')[0] if ' | ' in analysis.item_name else analysis.item_name
                        self.stats['weapon_stats'][weapon] = self.stats['weapon_stats'].get(weapon, 0) + 1
                        
                        if analysis.is_rare:
                            self.stats['rare_items_found'] += 1
                            self.stats['total_value_found'] += analysis.price
                            
                            # Update best find
                            if not self.stats['best_find'] or analysis.rarity_score > self.stats['best_find'].rarity_score:
                                self.stats['best_find'] = analysis
                            
                            self.logger.info(f"🚨 EXTREME FLOAT FOUND: {analysis.item_name} - Float: {analysis.float_value:.6f} - Score: {analysis.rarity_score}")
                            
                            # Send enhanced Telegram notification with real data
                            self.telegram.send_rare_float_alert(analysis)
                    else:
                        self.logger.debug(f"❌ Float {real_float:.6f} not extreme enough for {variant}")
                        continue
                
                self.logger.info(f"✅ Found {extreme_found} extreme floats in {variant}")
                
                # Reduced rate limiting since we're checking fewer items
                self.steam_api.rate_limit_delay()
        
        except Exception as e:
            self.logger.error(f"Error scanning {item_name}: {e}")
            self.stats['errors'] += 1
        
        return analyses
    
    def scan_item(self, item_name: str, max_listings: int = 100) -> List[FloatAnalysis]:
        """Legacy method - redirects to extreme float scanning"""
        return self.scan_item_extreme_floats(item_name, max_listings)
    
    async def _get_real_floats_batch(self, inspect_links: List[str]) -> List[Optional[FloatData]]:
        """Get real float values from CSFloat API for a batch of inspect links"""
        try:
            async with CSFloatAPI() as csfloat_api:
                self.logger.info(f"🔍 Requesting float data for {len(inspect_links)} items from CSFloat API")
                results = await csfloat_api.get_multiple_floats(inspect_links, max_concurrent=2)
                
                successful_count = sum(1 for r in results if r is not None)
                self.logger.info(f"✅ Successfully got {successful_count}/{len(inspect_links)} float values")
                
                return results
        except Exception as e:
            self.logger.error(f"💥 Error in CSFloat batch processing: {e}")
            # Fallback to individual requests
            return await self._get_real_floats_fallback(inspect_links)
    
    async def _get_real_floats_fallback(self, inspect_links: List[str]) -> List[Optional[FloatData]]:
        """Fallback method for getting floats one by one"""
        results = []
        self.logger.info("📋 Using fallback method for float extraction...")
        
        for i, link in enumerate(inspect_links):
            try:
                float_data = await get_float_data(link, use_fallback=True)
                results.append(float_data)
                
                if float_data:
                    self.logger.debug(f"✅ Fallback success {i+1}/{len(inspect_links)}: {float_data.float_value:.6f}")
                else:
                    self.logger.debug(f"❌ Fallback failed {i+1}/{len(inspect_links)}")
                
                # Small delay between requests
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in fallback for link {i+1}: {e}")
                results.append(None)
        
        return results
    
    def _estimate_price_from_float_data(self, float_data: FloatData) -> float:
        """Estimate market price based on float data and item information"""
        try:
            # Base price estimation based on weapon type and skin name
            base_price = 10.0  # Default minimum
            
            # Weapon type multipliers
            if 'Karambit' in float_data.weapon_type or 'Butterfly' in float_data.weapon_type:
                base_price = 300.0
            elif 'Bayonet' in float_data.weapon_type or 'M9' in float_data.weapon_type:
                base_price = 200.0
            elif 'AWP' in float_data.weapon_type:
                base_price = 50.0
            elif 'AK-47' in float_data.weapon_type:
                base_price = 30.0
            elif 'M4A4' in float_data.weapon_type or 'M4A1-S' in float_data.weapon_type:
                base_price = 25.0
            elif 'Glock' in float_data.weapon_type or 'USP' in float_data.weapon_type:
                base_price = 15.0
            
            # Skin rarity multipliers
            if 'Dragon Lore' in float_data.skin_name or 'Howl' in float_data.skin_name:
                base_price *= 10.0
            elif 'Fade' in float_data.skin_name or 'Doppler' in float_data.skin_name:
                base_price *= 3.0
            elif 'Asiimov' in float_data.skin_name or 'Redline' in float_data.skin_name:
                base_price *= 1.5
            
            # Float value premium/discount
            float_multiplier = 1.0
            
            if float_data.wear == 'Factory New':
                if float_data.float_value <= 0.001:
                    float_multiplier = 3.0  # 300% for extremely low floats
                elif float_data.float_value <= 0.005:
                    float_multiplier = 2.0  # 200% for very low floats
                elif float_data.float_value <= 0.01:
                    float_multiplier = 1.5  # 150% for low floats
            
            elif float_data.wear == 'Battle-Scarred':
                # Get skin-specific maximum for accurate premium calculation
                base_skin = float_data.market_name.split(' (')[0]
                skin_data = self.config.SKIN_SPECIFIC_RANGES.get(base_skin, {})
                bs_range = skin_data.get('Battle-Scarred', (0.45, 1.0))
                max_float = bs_range[1] if bs_range else 1.0
                
                # Calculate how close to maximum
                if max_float > 0:
                    closeness_to_max = float_data.float_value / max_float
                    if closeness_to_max >= 0.99:
                        float_multiplier = 2.5  # 250% for extremely high floats
                    elif closeness_to_max >= 0.95:
                        float_multiplier = 1.8  # 180% for very high floats
                    elif closeness_to_max >= 0.90:
                        float_multiplier = 1.3  # 130% for high floats
            
            estimated_price = base_price * float_multiplier
            
            return round(estimated_price, 2)
            
        except Exception as e:
            self.logger.error(f"Error estimating price from float data: {e}")
            return 50.0  # Safe default
    
    def simulate_extreme_float_value(self, item_name: str) -> Optional[float]:
        """Simulate extreme float values for testing (placeholder for real API)"""
        import random
        
        # Extract base skin name and wear condition
        base_skin = item_name.split(' (')[0] if ' (' in item_name else item_name
        wear_condition = "Factory New"  # Default
        for wear in self.config.WEAR_RANGES.keys():
            if wear in item_name:
                wear_condition = wear
                break
        
        # CRITICAL: Only allow Factory New and Battle-Scarred
        if wear_condition not in ["Factory New", "Battle-Scarred"]:
            self.logger.error(f"🚫 INVALID WEAR: {wear_condition} in {item_name} - Only FN and BS allowed!")
            return None
        
        # Get skin-specific ranges if available
        skin_data = self.config.SKIN_SPECIFIC_RANGES.get(base_skin, {})
        
        # Generate extreme floats with higher probability
        if wear_condition == "Factory New":
            # Check if this skin can exist in FN
            if skin_data.get('Factory New') is None and base_skin in self.config.WEAR_RESTRICTIONS['no_factory_new']:
                self.logger.error(f"🚫 {base_skin} cannot exist in Factory New!")
                return None
                
            extreme_fn = skin_data.get('extreme_fn', 0.0001)
            
            # 20% chance of generating extreme float
            if random.random() < 0.2:
                return random.uniform(0.0, extreme_fn)
            else:
                # Normal FN range 
                fn_range = skin_data.get('Factory New', (0.00, 0.07))
                if fn_range:
                    return random.uniform(fn_range[0], fn_range[1])
                return random.uniform(0.0, 0.07)
                
        elif wear_condition == "Battle-Scarred":
            # Check if this skin can exist in BS
            if skin_data.get('Battle-Scarred') is None and base_skin in self.config.WEAR_RESTRICTIONS['no_battle_scarred']:
                self.logger.error(f"🚫 {base_skin} cannot exist in Battle-Scarred!")
                return None
                
            # Get skin-specific BS range
            bs_range = skin_data.get('Battle-Scarred', (0.45, 1.00))
            extreme_bs = skin_data.get('extreme_bs', 0.999)
            
            if bs_range:
                # 20% chance of generating extreme float
                if random.random() < 0.2:
                    return random.uniform(extreme_bs, bs_range[1])
                else:
                    # Normal BS range
                    return random.uniform(bs_range[0], bs_range[1])
            else:
                return None  # This wear doesn't exist for this skin
        
        return None
    
    def simulate_float_value(self, item_name: str) -> Optional[float]:
        """Simulate float value extraction (placeholder) - legacy method"""
        # Redirect to extreme float simulation for better testing
        return self.simulate_extreme_float_value(item_name)
    
    def scan_multiple_items(self, item_names: List[str], max_listings_per_item: int = 50) -> Dict[str, List[FloatAnalysis]]:
        """Scan multiple items for EXTREME FLOATS ONLY (FN < 0.0001, BS > 0.99)"""
        self.stats['start_time'] = datetime.now()
        results = {}
        
        self.logger.info(f"🎯 Starting EXTREME FLOAT scan of {len(item_names)} items...")
        self.logger.info("🔍 Target: Factory New < 0.0001 and Battle-Scarred > 0.99 ONLY")
        self.telegram.send_startup_notification()
        
        total_variants_to_scan = 0
        for item_name in item_names:
            variants = self.steam_api.get_extreme_float_variants(item_name)
            total_variants_to_scan += len(variants)
        
        self.logger.info(f"📊 Will scan {total_variants_to_scan} extreme float variants (instead of {len(item_names) * 5} with old method)")
        
        for i, item_name in enumerate(item_names, 1):
            self.logger.info(f"🎯 Progress: {i}/{len(item_names)} - {item_name}")
            
            # Use extreme float scanning only
            analyses = self.scan_item_extreme_floats(item_name, max_listings_per_item)
            results[item_name] = analyses
            
            # Print progress
            self.print_progress_stats()
        
        self.logger.info("🎯 EXTREME FLOAT scan completed!")
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
        """Scan all available weapons from the database for EXTREME FLOATS ONLY"""
        self.logger.info("🎯 Starting comprehensive weapon scan for EXTREME FLOATS...")
        self.logger.info("🔍 Target: Only Factory New < 0.0001 and Battle-Scarred > 0.99")
        
        all_weapons = self.skin_db.get_all_weapons()
        self.logger.info(f"Found {len(all_weapons)} weapon types in database")
        
        results = {}
        extreme_float_items = []
        
        # Get representative skins for each weapon that can have extreme floats
        for weapon in all_weapons[:50]:  # Limit to first 50 weapons to avoid overwhelming
            weapon_skins = self.skin_db.get_skins_by_weapon(weapon)
            
            # Select most popular/valuable skins for each weapon
            selected_skins = weapon_skins[:max_items_per_weapon]
            
            for skin in selected_skins:
                # Only add skins that can actually have extreme floats
                variants = self.steam_api.get_extreme_float_variants(skin.name)
                if variants:  # Only add if it has valid FN or BS variants
                    extreme_float_items.append(skin.name)
        
        self.logger.info(f"🎯 Selected {len(extreme_float_items)} items that can have extreme floats")
        self.logger.info("⚠️ Skipped items that can't have FN < 0.0001 or BS > 0.99")
        
        # Use extreme float scanning for all items
        return self.scan_multiple_items(extreme_float_items)
    
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