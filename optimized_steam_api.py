import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from urllib.parse import quote
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

from config import FloatCheckerConfig

@dataclass
class RateLimitTracker:
    requests_made: int = 0
    window_start: float = 0
    current_delay: float = 0.15
    consecutive_errors: int = 0
    last_request_time: float = 0

class OptimizedSteamAPI:
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        self.rate_tracker = RateLimitTracker()
        self.session = None
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'rate_limited_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'requests_per_second': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.config.CONNECTION_POOL_SIZE,
            limit_per_host=self.config.CONNECTION_POOL_SIZE,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.TIMEOUT)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _adaptive_delay(self):
        """Implement adaptive rate limiting"""
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.rate_tracker.window_start >= 1.0:
            self.rate_tracker.requests_made = 0
            self.rate_tracker.window_start = current_time
        
        # Check if we need to slow down
        if self.rate_tracker.requests_made >= self.config.REQUESTS_PER_SECOND_LIMIT:
            sleep_time = 1.0 - (current_time - self.rate_tracker.window_start)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                self.rate_tracker.window_start = time.time()
                self.rate_tracker.requests_made = 0
        
        # Adaptive delay based on recent performance
        if self.config.ADAPTIVE_RATE_LIMITING:
            if self.rate_tracker.consecutive_errors > 3:
                self.rate_tracker.current_delay = min(
                    self.rate_tracker.current_delay * 1.5,
                    self.config.EXPONENTIAL_BACKOFF['max_delay']
                )
            elif self.rate_tracker.consecutive_errors == 0:
                self.rate_tracker.current_delay = max(
                    self.rate_tracker.current_delay * 0.9,
                    self.config.BURST_REQUEST_DELAY
                )
        
        # Minimum delay between requests
        time_since_last = current_time - self.rate_tracker.last_request_time
        if time_since_last < self.rate_tracker.current_delay:
            await asyncio.sleep(self.rate_tracker.current_delay - time_since_last)
        
        self.rate_tracker.last_request_time = time.time()
        self.rate_tracker.requests_made += 1
    
    async def _make_request(self, url: str, params: Dict = None, retries: int = 0) -> Optional[Dict]:
        """Make a rate-limited request with error handling"""
        await self._adaptive_delay()
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            async with self.session.get(url, params=params) as response:
                response_time = time.time() - start_time
                
                # Update performance stats
                self._update_performance_stats(response_time)
                
                if response.status == 429:  # Rate limited
                    self.stats['rate_limited_requests'] += 1
                    self.rate_tracker.consecutive_errors += 1
                    
                    if retries < self.config.MAX_RETRIES:
                        backoff_delay = self.config.EXPONENTIAL_BACKOFF['initial_delay'] * (
                            self.config.EXPONENTIAL_BACKOFF['multiplier'] ** retries
                        )
                        backoff_delay = min(backoff_delay, self.config.EXPONENTIAL_BACKOFF['max_delay'])
                        
                        self.logger.warning(f"Rate limited, retrying in {backoff_delay:.1f}s (attempt {retries + 1})")
                        await asyncio.sleep(backoff_delay)
                        return await self._make_request(url, params, retries + 1)
                    else:
                        self.logger.error("Max retries exceeded for rate limiting")
                        return None
                
                elif response.status == 200:
                    self.stats['successful_requests'] += 1
                    self.rate_tracker.consecutive_errors = 0
                    
                    try:
                        return await response.json()
                    except json.JSONDecodeError:
                        self.logger.error("Failed to decode JSON response")
                        return None
                
                else:
                    self.stats['failed_requests'] += 1
                    self.rate_tracker.consecutive_errors += 1
                    self.logger.error(f"HTTP {response.status} error for {url}")
                    return None
        
        except asyncio.TimeoutError:
            self.stats['failed_requests'] += 1
            self.rate_tracker.consecutive_errors += 1
            self.logger.error(f"Timeout for {url}")
            return None
        
        except Exception as e:
            self.stats['failed_requests'] += 1
            self.rate_tracker.consecutive_errors += 1
            self.logger.error(f"Request error for {url}: {e}")
            return None
    
    def _update_performance_stats(self, response_time: float):
        """Update performance statistics"""
        # Update average response time
        total_successful = self.stats['successful_requests']
        if total_successful > 0:
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (total_successful - 1) + response_time) / total_successful
            )
        
        # Calculate requests per second
        if self.stats['total_requests'] > 0:
            elapsed = time.time() - getattr(self, '_start_time', time.time())
            if elapsed > 0:
                self.stats['requests_per_second'] = self.stats['total_requests'] / elapsed
    
    async def search_items_batch(self, search_terms: List[str], count: int = 100) -> AsyncGenerator[Tuple[str, Dict], None]:
        """Search for multiple items concurrently"""
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        
        semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        async def search_single(term: str) -> Tuple[str, Optional[Dict]]:
            async with semaphore:
                result = await self.search_items(term, count=count)
                return term, result
        
        # Create tasks for all search terms
        tasks = [search_single(term) for term in search_terms]
        
        # Process tasks in batches to avoid overwhelming the system
        batch_size = self.config.MAX_CONCURRENT_REQUESTS * 2
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch search error: {result}")
                    continue
                
                term, data = result
                if data:
                    yield term, data
    
    async def search_items(self, search_term: str, start: int = 0, count: int = 100) -> Optional[Dict]:
        """Search for items on Steam Market"""
        params = {
            'query': search_term,
            'start': start,
            'count': count,
            'search_descriptions': 0,
            'sort_column': 'popular',
            'sort_dir': 'desc',
            'appid': self.config.CS2_APP_ID,
            'category_730_ItemSet[]': 'any',
            'category_730_ProPlayer[]': 'any',
            'category_730_StickerCapsule[]': 'any',
            'category_730_TournamentTeam[]': 'any',
            'category_730_Weapon[]': 'any'
        }
        
        return await self._make_request(self.config.STEAM_MARKET_SEARCH_URL, params)
    
    async def get_item_listings_batch(self, item_names: List[str], count: int = 100) -> AsyncGenerator[Tuple[str, Dict], None]:
        """Get listings for multiple items concurrently"""
        semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        async def get_single_listing(item_name: str) -> Tuple[str, Optional[Dict]]:
            async with semaphore:
                result = await self.get_item_listings(item_name, count=count)
                return item_name, result
        
        # Process in batches
        batch_size = self.config.MAX_CONCURRENT_REQUESTS * 2
        tasks = [get_single_listing(name) for name in item_names]
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch listing error: {result}")
                    continue
                
                item_name, data = result
                if data:
                    yield item_name, data
    
    async def get_item_listings(self, market_hash_name: str, start: int = 0, count: int = 100) -> Optional[Dict]:
        """Get listings for a specific item"""
        encoded_name = quote(market_hash_name)
        url = f"{self.config.STEAM_MARKET_LISTINGS_URL}/{encoded_name}/render/"
        
        params = {
            'query': '',
            'start': start,
            'count': count,
            'country': 'US',
            'language': 'english',
            'currency': 1
        }
        
        return await self._make_request(url, params)
    
    async def get_price_history(self, market_hash_name: str) -> Optional[Dict]:
        """Get price history for an item"""
        url = f"{self.config.STEAM_MARKET_BASE_URL}/pricehistory/"
        
        params = {
            'appid': self.config.CS2_APP_ID,
            'market_hash_name': market_hash_name
        }
        
        return await self._make_request(url, params)
    
    def extract_inspect_links(self, listings_data: Dict) -> List[str]:
        """Extract inspect links from listings data"""
        inspect_links = []
        
        if 'listinginfo' in listings_data:
            for listing_id, listing_info in listings_data['listinginfo'].items():
                if 'asset' in listing_info:
                    asset = listing_info['asset']
                    if 'market_actions' in asset:
                        for action in asset['market_actions']:
                            if 'link' in action and 'inspect' in action['link'].lower():
                                inspect_links.append(action['link'])
        
        return inspect_links
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        success_rate = 0
        if self.stats['total_requests'] > 0:
            success_rate = (self.stats['successful_requests'] / self.stats['total_requests']) * 100
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'current_delay': self.rate_tracker.current_delay,
            'consecutive_errors': self.rate_tracker.consecutive_errors
        }
    
    async def scan_entire_market(self, batch_size: int = 50) -> AsyncGenerator[Tuple[str, Dict], None]:
        """Scan the entire CS2 market efficiently"""
        self.logger.info("Starting comprehensive market scan...")
        
        # First, get all possible items from a broad search
        broad_searches = [
            "AK-47", "M4A4", "M4A1-S", "AWP", "Glock-18", "USP-S", "Desert Eagle",
            "Karambit", "M9 Bayonet", "Butterfly Knife", "Bayonet", "Flip Knife",
            "Falchion Knife", "Shadow Daggers", "Bowie Knife", "Huntsman Knife",
            "Gut Knife", "Ursus Knife", "Navaja Knife", "Stiletto Knife",
            "Talon Knife", "Classic Knife", "Paracord Knife", "Survival Knife",
            "Nomad Knife", "Skeleton Knife", "Specialist Gloves", "Sport Gloves",
            "Bloodhound Gloves", "Driver Gloves", "Hand Wraps", "Moto Gloves",
            "Hydra Gloves", "Broken Fang Gloves"
        ]
        
        # Add weapon categories
        weapons = [
            "P2000", "Five-SeveN", "Tec-9", "CZ75-Auto", "Dual Berettas", "P250",
            "R8 Revolver", "MP9", "MAC-10", "PP-Bizon", "UMP-45", "P90", "MP5-SD",
            "MP7", "Galil AR", "FAMAS", "AUG", "SG 553", "SSG 08", "SCAR-20",
            "G3SG1", "Nova", "XM1014", "Sawed-Off", "MAG-7", "M249", "Negev"
        ]
        
        all_search_terms = broad_searches + weapons
        
        processed_items = set()
        
        async for search_term, search_results in self.search_items_batch(all_search_terms, count=100):
            if not search_results or 'results_html' not in search_results:
                continue
            
            # Parse items from search results
            # This is a simplified version - you'd need to parse the HTML properly
            items_found = self._extract_items_from_search(search_results)
            
            for item_name in items_found:
                if item_name not in processed_items:
                    processed_items.add(item_name)
                    
                    # Get detailed listings for this item
                    listings = await self.get_item_listings(item_name, count=batch_size)
                    if listings:
                        yield item_name, listings
        
        self.logger.info(f"Market scan completed. Processed {len(processed_items)} unique items.")
    
    def _extract_items_from_search(self, search_results: Dict) -> List[str]:
        """Extract item names from search results HTML"""
        # This is a placeholder - in a real implementation, you'd parse the HTML
        # to extract actual item names from the search results
        # For now, return empty list as we'd need proper HTML parsing
        return []