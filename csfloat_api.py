#!/usr/bin/env python3

import aiohttp
import asyncio
import logging
import time
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import quote, unquote
from datetime import datetime, timedelta

@dataclass
class FloatData:
    """Data structure for CSFloat API response"""
    item_name: str
    float_value: float
    paint_seed: int
    paint_index: int
    wear: str
    low_rank: int
    high_rank: int
    image_url: str
    inspect_link: str
    market_name: str
    weapon_type: str
    skin_name: str
    rarity: str
    stickers: List[Dict] = None
    
    def __post_init__(self):
        if self.stickers is None:
            self.stickers = []

class CSFloatAPI:
    """CSFloat API client for accurate float value extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://api.csfloat.com"
        self.session = None
        
        # Rate limiting
        self.rate_limit = 10  # requests per minute
        self.request_times = []
        self.min_request_interval = 6  # seconds between requests
        self.last_request_time = 0
        
        # Error handling
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_reset_time = None
        
        # Cache
        self.cache = {}
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'SteamBot-FloatChecker/1.0',
                'Accept': 'application/json'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            if self.circuit_breaker_reset_time and datetime.now() > self.circuit_breaker_reset_time:
                # Reset circuit breaker
                self.circuit_breaker_failures = 0
                self.circuit_breaker_reset_time = None
                self.logger.info("🔄 Circuit breaker reset - resuming CSFloat API calls")
                return False
            return True
        return False
    
    async def _rate_limit_delay(self):
        """Apply rate limiting delay"""
        current_time = time.time()
        
        # Clean old request times
        cutoff_time = current_time - 60  # 1 minute ago
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check if we're hitting rate limits
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.info(f"⏱️ Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.request_times.append(time.time())
        self.last_request_time = time.time()
    
    def _extract_inspect_params(self, inspect_link: str) -> Optional[Dict[str, str]]:
        """Extract parameters from Steam inspect link"""
        try:
            # Handle both encoded and decoded inspect links
            if '%20' in inspect_link:
                inspect_link = unquote(inspect_link)
            
            # Extract S, A, D, M parameters
            pattern = r'S(\d+)A(\d+)D(\d+)(?:M(\d+))?'
            match = re.search(pattern, inspect_link)
            
            if match:
                params = {
                    's': match.group(1),  # Steam ID
                    'a': match.group(2),  # Asset ID  
                    'd': match.group(3),  # D parameter
                }
                
                # M parameter is optional
                if match.group(4):
                    params['m'] = match.group(4)
                
                self.logger.debug(f"Extracted inspect params: {params}")
                return params
            else:
                self.logger.error(f"Could not parse inspect link: {inspect_link}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting inspect params: {e}")
            return None
    
    def _get_cache_key(self, inspect_link: str) -> str:
        """Generate cache key for inspect link"""
        params = self._extract_inspect_params(inspect_link)
        if params:
            return f"{params.get('s', '')}_{params.get('a', '')}_{params.get('d', '')}"
        return inspect_link
    
    def _is_cached(self, cache_key: str) -> Tuple[bool, Optional[FloatData]]:
        """Check if result is cached and still valid"""
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return True, cached_data
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        return False, None
    
    async def get_float_value(self, inspect_link: str) -> Optional[FloatData]:
        """Get float value from CSFloat API"""
        if self._is_circuit_breaker_open():
            self.logger.warning("🚫 Circuit breaker open - skipping CSFloat API call")
            return None
        
        cache_key = self._get_cache_key(inspect_link)
        is_cached, cached_data = self._is_cached(cache_key)
        
        if is_cached:
            self.logger.debug(f"📋 Using cached data for {cache_key}")
            return cached_data
        
        try:
            await self._rate_limit_delay()
            
            # Extract parameters for API call
            params = self._extract_inspect_params(inspect_link)
            if not params:
                return None
            
            # Make API request
            url = f"{self.base_url}/"
            query_params = {
                'url': inspect_link
            }
            
            self.logger.debug(f"🔍 Requesting float data from CSFloat API")
            
            async with self.session.get(url, params=query_params) as response:
                if response.status == 200:
                    data = await response.json()
                    float_data = self._parse_csfloat_response(data, inspect_link)
                    
                    if float_data:
                        # Cache successful result
                        self.cache[cache_key] = (float_data, datetime.now())
                        self.circuit_breaker_failures = 0  # Reset on success
                        
                        self.logger.info(f"✅ Got float data: {float_data.item_name} - {float_data.float_value:.6f}")
                        return float_data
                    else:
                        self.logger.warning("⚠️ CSFloat API returned invalid data")
                        return None
                
                elif response.status == 429:
                    # Rate limited
                    self.logger.warning("⏱️ Rate limited by CSFloat API")
                    await asyncio.sleep(60)  # Wait 1 minute
                    return None
                
                elif response.status in [500, 502, 503, 504]:
                    # Server error
                    self.logger.error(f"🔥 CSFloat API server error: {response.status}")
                    self._handle_api_failure()
                    return None
                
                else:
                    self.logger.error(f"🚫 CSFloat API error: {response.status}")
                    self._handle_api_failure()
                    return None
                    
        except asyncio.TimeoutError:
            self.logger.error("⏰ CSFloat API request timeout")
            self._handle_api_failure()
            return None
            
        except Exception as e:
            self.logger.error(f"💥 CSFloat API error: {e}")
            self._handle_api_failure()
            return None
    
    def _parse_csfloat_response(self, data: Dict, inspect_link: str) -> Optional[FloatData]:
        """Parse CSFloat API response into FloatData object"""
        try:
            if 'iteminfo' not in data:
                self.logger.error("CSFloat response missing iteminfo")
                return None
            
            item_info = data['iteminfo']
            
            # Extract basic float information
            float_value = item_info.get('floatvalue', 0.0)
            paint_seed = item_info.get('paintseed', 0)
            paint_index = item_info.get('paintindex', 0)
            
            # Extract item name information
            full_name = item_info.get('full_item_name', '')
            market_name = item_info.get('market_name', full_name)
            
            # Parse weapon and skin name
            weapon_type = ""
            skin_name = ""
            if ' | ' in market_name:
                weapon_type, skin_name = market_name.split(' | ', 1)
                # Remove wear condition from skin name if present
                if ' (' in skin_name:
                    skin_name = skin_name.split(' (')[0]
            
            # Extract wear condition
            wear = item_info.get('wear_name', '')
            
            # Extract ranking information
            low_rank = item_info.get('low_rank', 0)
            high_rank = item_info.get('high_rank', 0)
            
            # Extract rarity
            rarity = item_info.get('rarity', '')
            
            # Extract image URL
            image_url = item_info.get('imageurl', '')
            
            # Extract stickers
            stickers = []
            if 'stickers' in item_info:
                for sticker_data in item_info['stickers']:
                    stickers.append({
                        'name': sticker_data.get('name', ''),
                        'wear': sticker_data.get('wear', 0.0),
                        'slot': sticker_data.get('slot', 0)
                    })
            
            return FloatData(
                item_name=market_name,
                float_value=float_value,
                paint_seed=paint_seed,
                paint_index=paint_index,
                wear=wear,
                low_rank=low_rank,
                high_rank=high_rank,
                image_url=image_url,
                inspect_link=inspect_link,
                market_name=market_name,
                weapon_type=weapon_type,
                skin_name=skin_name,
                rarity=rarity,
                stickers=stickers
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing CSFloat response: {e}")
            return None
    
    def _handle_api_failure(self):
        """Handle API failure for circuit breaker"""
        self.circuit_breaker_failures += 1
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_reset_time = datetime.now() + timedelta(minutes=10)
            self.logger.error(f"🚫 Circuit breaker opened after {self.circuit_breaker_failures} failures")
    
    async def get_multiple_floats(self, inspect_links: List[str], 
                                max_concurrent: int = 3) -> List[Optional[FloatData]]:
        """Get float values for multiple inspect links with concurrency control"""
        if self._is_circuit_breaker_open():
            self.logger.warning("🚫 Circuit breaker open - skipping batch float requests")
            return [None] * len(inspect_links)
        
        results = []
        
        # Process in batches to respect rate limits
        for i in range(0, len(inspect_links), max_concurrent):
            batch = inspect_links[i:i + max_concurrent]
            
            # Create tasks for concurrent processing
            tasks = [self.get_float_value(link) for link in batch]
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error in batch processing: {result}")
                    results.append(None)
                else:
                    results.append(result)
            
            # Delay between batches
            if i + max_concurrent < len(inspect_links):
                await asyncio.sleep(2)
        
        return results
    
    def get_api_status(self) -> Dict[str, any]:
        """Get current API status and statistics"""
        return {
            'circuit_breaker_open': self._is_circuit_breaker_open(),
            'failures': self.circuit_breaker_failures,
            'cache_size': len(self.cache),
            'recent_requests': len(self.request_times),
            'rate_limit': self.rate_limit,
            'last_request': self.last_request_time
        }

# Alternative API fallback
class TradeitAPI:
    """Fallback API for float checking when CSFloat is unavailable"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://tradeit.gg/api"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_float_value(self, inspect_link: str) -> Optional[FloatData]:
        """Fallback float value extraction (placeholder)"""
        # This would implement Tradeit.gg API integration
        # For now, return None to indicate fallback not available
        self.logger.info("📋 Tradeit API fallback not yet implemented")
        return None

# Factory function
async def get_float_data(inspect_link: str, use_fallback: bool = True) -> Optional[FloatData]:
    """Factory function to get float data with fallback support"""
    
    # Try CSFloat API first
    try:
        async with CSFloatAPI() as csfloat:
            result = await csfloat.get_float_value(inspect_link)
            if result:
                return result
    except Exception as e:
        logging.error(f"CSFloat API failed: {e}")
    
    # Try fallback API if enabled
    if use_fallback:
        try:
            async with TradeitAPI() as tradeit:
                result = await tradeit.get_float_value(inspect_link)
                if result:
                    return result
        except Exception as e:
            logging.error(f"Fallback API failed: {e}")
    
    return None

# Test function
async def test_csfloat_api():
    """Test CSFloat API functionality"""
    print("🧪 Testing CSFloat API...")
    
    # Test inspect link (this would be a real one in practice)
    test_inspect_link = "steam://rungame/730/76561202255233023/+csgo_econ_action_preview%20S76561198084749846A123456789D456789123456789"
    
    async with CSFloatAPI() as api:
        # Test single request
        print("Testing single float request...")
        float_data = await api.get_float_value(test_inspect_link)
        
        if float_data:
            print(f"✅ Success: {float_data.item_name} - Float: {float_data.float_value:.6f}")
        else:
            print("⚠️ No data returned (expected for test link)")
        
        # Test API status
        status = api.get_api_status()
        print(f"📊 API Status: {status}")
        
        print("✅ CSFloat API test completed")

if __name__ == "__main__":
    asyncio.run(test_csfloat_api())