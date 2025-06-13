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
from csfloat_api import CSFloatAPI, FloatData, get_float_data

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
    
    async def scan_extreme_floats_optimized(self) -> Dict[str, List[FloatAnalysis]]:
        """Scan ONLY for extreme float values (FN < 0.0001 and BS > 0.99) with maximum efficiency"""
        self.logger.info("🎯 Starting optimized EXTREME FLOAT ONLY scan...")
        self.logger.info("🔍 Target: Factory New < 0.0001 and Battle-Scarred > 0.99")
        self.telegram.send_startup_notification()
        
        results = {}
        scan_start_time = time.time()
        
        async with OptimizedSteamAPI() as steam_api:
            # Get prioritized list of items that can have extreme floats
            extreme_targets = self._get_extreme_float_targets()
            total_variants = len(extreme_targets)
            
            self.logger.info(f"🎯 Scanning {total_variants} extreme float variants (reduced from ~10,000+ variants)")
            
            # Process items in smaller, focused batches
            batch_size = self.config.MAX_CONCURRENT_REQUESTS * 2  # Smaller batches for precision
            
            for i in range(0, total_variants, batch_size):
                if self.should_stop:
                    break
                
                batch = extreme_targets[i:i + batch_size]
                batch_results = await self._process_extreme_float_batch(steam_api, batch)
                
                for item_name, analyses in batch_results.items():
                    if analyses:
                        results[item_name] = analyses
                
                # Update progress
                progress = ((i + batch_size) / total_variants) * 100
                self._update_scanning_stats(steam_api, progress)
                
                # Brief pause between batches
                await asyncio.sleep(0.3)
        
        scan_duration = time.time() - scan_start_time
        self._log_extreme_float_results(scan_duration, results)
        
        return results
    
    async def scan_entire_market_optimized(self) -> Dict[str, List[FloatAnalysis]]:
        """Legacy method - redirects to extreme float scanning for efficiency"""
        return await self.scan_extreme_floats_optimized()
    
    def _get_extreme_float_targets(self) -> List[str]:
        """Get list of skin variants that can have extreme floats"""
        targets = []
        
        # Get base skins from skin database
        all_weapons = self.skin_db.get_all_weapons()
        
        for weapon in all_weapons[:100]:  # Limit to top 100 weapons for performance
            weapon_skins = self.skin_db.get_skins_by_weapon(weapon)
            
            for skin in weapon_skins[:20]:  # Top 20 skins per weapon
                base_name = skin.name
                
                # Check if this skin exists in our specific ranges database
                if base_name in self.config.SKIN_SPECIFIC_RANGES:
                    skin_data = self.config.SKIN_SPECIFIC_RANGES[base_name]
                    
                    # Add Factory New variant if it exists and can have extreme floats
                    if skin_data.get('extreme_fn') is not None:
                        if base_name not in self.config.WEAR_RESTRICTIONS['no_factory_new']:
                            targets.append(f"{base_name} (Factory New)")
                    
                    # Add Battle-Scarred variant if it exists and can have extreme floats
                    if skin_data.get('extreme_bs') is not None:
                        if base_name not in self.config.WEAR_RESTRICTIONS['no_battle_scarred']:
                            targets.append(f"{base_name} (Battle-Scarred)")
                
                # For skins not in our database, use generic extreme thresholds
                else:
                    # Add FN if skin can exist in FN
                    if base_name not in self.config.WEAR_RESTRICTIONS['no_factory_new']:
                        targets.append(f"{base_name} (Factory New)")
                    
                    # Add BS if skin can exist in BS
                    if base_name not in self.config.WEAR_RESTRICTIONS['no_battle_scarred']:
                        targets.append(f"{base_name} (Battle-Scarred)")
        
        # Add high-priority items from monitored list
        for monitored_item in self.config.MONITORED_ITEMS:
            if monitored_item not in self.config.WEAR_RESTRICTIONS['no_factory_new']:
                fn_variant = f"{monitored_item} (Factory New)"
                if fn_variant not in targets:
                    targets.append(fn_variant)
            
            if monitored_item not in self.config.WEAR_RESTRICTIONS['no_battle_scarred']:
                bs_variant = f"{monitored_item} (Battle-Scarred)"
                if bs_variant not in targets:
                    targets.append(bs_variant)
        
        self.logger.info(f"🎯 Generated {len(targets)} extreme float target variants")
        return targets
    
    async def _process_extreme_float_batch(self, steam_api, item_variants: List[str]) -> Dict[str, List[FloatAnalysis]]:
        """Process a batch of item variants looking only for extreme floats"""
        results = {}
        
        # Create tasks for concurrent processing
        tasks = []
        for variant in item_variants:
            task = self._scan_single_variant_for_extremes(steam_api, variant)
            tasks.append(task)
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing {item_variants[i]}: {result}")
                continue
            
            if result:
                variant_name = item_variants[i]
                results[variant_name] = result
        
        return results
    
    async def _scan_single_variant_for_extremes(self, steam_api, variant_name: str) -> List[FloatAnalysis]:
        """Scan a single variant for extreme floats only"""
        try:
            # Get market listings for this specific variant
            listings_data = await steam_api.get_item_listings_async(variant_name, count=50)
            
            if not listings_data:
                return []
            
            # Extract inspect links
            inspect_links = steam_api.extract_inspect_links(listings_data)
            
            extreme_analyses = []
            
            # Process inspect links with REAL CSFloat API
            self.logger.info(f"🔍 Processing {len(inspect_links)} inspect links with CSFloat API...")
            
            # Get real float values in batch
            try:
                async with CSFloatAPI() as csfloat_api:
                    real_floats = await csfloat_api.get_multiple_floats(inspect_links[:10], max_concurrent=2)
                    
                    for i, (inspect_link, float_data) in enumerate(zip(inspect_links[:10], real_floats)):
                        if float_data is None:
                            continue
                        
                        # Use REAL float value
                        real_float = float_data.float_value
                        
                        # Check if this is truly an extreme float
                        if self._is_extreme_float(real_float, variant_name):
                            # Get estimated price based on real float data
                            estimated_price = self._estimate_price_from_float_data(float_data)
                            
                            # Analyze the float
                            analysis = self.analyzer.analyze_float_rarity(
                                variant_name, real_float, estimated_price
                            )
                            analysis.inspect_link = inspect_link
                            
                            # Add CSFloat-specific data
                            analysis.paint_seed = float_data.paint_seed
                            analysis.paint_index = float_data.paint_index
                            analysis.rank_info = f"Rank {float_data.low_rank}-{float_data.high_rank}"
                            
                            # Only keep truly rare/extreme items
                            if analysis.rarity_score >= 80:  # High rarity threshold
                                extreme_analyses.append(analysis)
                                
                                # Save to database
                                self.database.save_analysis(analysis)
                                
                                # Update stats
                                with self.performance_lock:
                                    self.stats.items_scanned += 1
                                    if analysis.is_rare:
                                        self.stats.rare_items_found += 1
                                        self.stats.total_value_found += analysis.price
                                
                                self.logger.info(f"🚨 EXTREME FLOAT: {variant_name} - {real_float:.6f} - Score: {analysis.rarity_score}")
                        else:
                            self.logger.debug(f"❌ Float {real_float:.6f} not extreme for {variant_name}")
                        
                        # Small delay between items
                        await asyncio.sleep(0.1)
                        
            except Exception as e:
                self.logger.error(f"Error in CSFloat processing: {e}")
                # Fallback to individual requests if batch fails
                for inspect_link in inspect_links[:5]:  # Reduced count for fallback
                    try:
                        float_data = await get_float_data(inspect_link)
                        if float_data and self._is_extreme_float(float_data.float_value, variant_name):
                            # Process as above but individually
                            analysis = self.analyzer.analyze_float_rarity(
                                variant_name, float_data.float_value, 
                                self._estimate_price_from_float_data(float_data)
                            )
                            analysis.inspect_link = inspect_link
                            
                            if analysis.rarity_score >= 80:
                                extreme_analyses.append(analysis)
                                self.database.save_analysis(analysis)
                                
                                with self.performance_lock:
                                    self.stats.items_scanned += 1
                                    if analysis.is_rare:
                                        self.stats.rare_items_found += 1
                                        self.stats.total_value_found += analysis.price
                    except Exception as inner_e:
                        self.logger.error(f"Error in fallback processing: {inner_e}")
                        continue
                    
                except Exception as e:
                    self.logger.debug(f"Error processing inspect link: {e}")
                    continue
            
            return extreme_analyses
            
        except Exception as e:
            self.logger.error(f"Error scanning {variant_name}: {e}")
            return []
    
    def _simulate_extreme_float_for_variant(self, variant_name: str) -> Optional[float]:
        """Simulate extreme float values with realistic probability"""
        import random
        
        base_skin = variant_name.split(' (')[0]
        
        # Get skin-specific data
        skin_data = self.config.SKIN_SPECIFIC_RANGES.get(base_skin, {})
        
        if 'Factory New' in variant_name:
            extreme_threshold = skin_data.get('extreme_fn', 0.0001)
            # 10% chance of extreme float in simulation
            if random.random() < 0.1:
                return random.uniform(0.0, extreme_threshold)
            else:
                # Normal FN range
                fn_range = skin_data.get('Factory New', (0.00, 0.07))
                return random.uniform(fn_range[0], fn_range[1]) if fn_range else random.uniform(0.0, 0.07)
        
        elif 'Battle-Scarred' in variant_name:
            bs_range = skin_data.get('Battle-Scarred', (0.45, 1.00))
            extreme_threshold = skin_data.get('extreme_bs', 0.999)
            
            if bs_range:
                # 10% chance of extreme float
                if random.random() < 0.1:
                    return random.uniform(extreme_threshold, bs_range[1])
                else:
                    return random.uniform(bs_range[0], bs_range[1])
        
        return None
    
    def _is_extreme_float(self, float_value: float, variant_name: str) -> bool:
        """Check if a float value qualifies as extreme for the specific variant"""
        base_skin = variant_name.split(' (')[0]
        skin_data = self.config.SKIN_SPECIFIC_RANGES.get(base_skin, {})
        
        if 'Factory New' in variant_name:
            extreme_fn = skin_data.get('extreme_fn', 0.0001)
            return float_value <= extreme_fn
        
        elif 'Battle-Scarred' in variant_name:
            extreme_bs = skin_data.get('extreme_bs', 0.999)
            return float_value >= extreme_bs
        
        return False
    
    def _estimate_extreme_float_price(self, variant_name: str, float_value: float) -> float:
        """Estimate price for extreme float items (they're worth more)"""
        # Base price simulation
        base_price = 50.0  # Minimum for extreme floats
        
        if 'Karambit' in variant_name or 'Butterfly' in variant_name:
            base_price = 500.0
        elif 'Dragon Lore' in variant_name or 'Howl' in variant_name:
            base_price = 2000.0
        elif 'AK-47' in variant_name or 'AWP' in variant_name:
            base_price = 100.0
        
        # Extreme float premium
        if 'Factory New' in variant_name and float_value <= 0.0001:
            base_price *= 2.0  # 100% premium for 0.000x floats
        elif 'Battle-Scarred' in variant_name and float_value >= 0.999:
            base_price *= 1.5  # 50% premium for 0.999+ floats
        
        return base_price
    
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
    
    def _log_extreme_float_results(self, scan_duration: float, results: Dict):
        """Log final results for extreme float scan"""
        total_extreme_found = sum(len(analyses) for analyses in results.values())
        
        self.logger.info(f"🎯 EXTREME FLOAT SCAN COMPLETED")
        self.logger.info(f"⏱️ Duration: {scan_duration:.1f} seconds")
        self.logger.info(f"🚨 Extreme floats found: {total_extreme_found}")
        self.logger.info(f"💰 Total value of extreme items: ${self.stats.total_value_found:.2f}")
        self.logger.info(f"📊 Items scanned: {self.stats.items_scanned}")
        self.logger.info(f"⚡ Scan rate: {self.stats.items_scanned / scan_duration:.1f} items/sec")
        
        if self.stats.best_find:
            self.logger.info(f"🏆 Best find: {self.stats.best_find.item_name} - {self.stats.best_find.float_value:.6f}")
    
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