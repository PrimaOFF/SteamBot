import os
import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import time

try:
    from steampy.client import SteamClient
    from steampy.utils import GameOptions
    from steampy.models import Currency
    STEAMPY_AVAILABLE = True
except ImportError:
    STEAMPY_AVAILABLE = False
    logging.warning("steampy not available - install with: pip install steampy")

from config import FloatCheckerConfig

class SteamAuthenticator:
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.session_file = "steam_session.json"
        self.credentials_file = "steam_credentials.json"
        
        # Steam credentials
        self.username = None
        self.password = None
        self.shared_secret = None
        self.identity_secret = None
        
        self._load_credentials()
    
    def _load_credentials(self):
        """Load Steam credentials from secure storage"""
        try:
            if os.path.exists(self.credentials_file):
                with open(self.credentials_file, 'r') as f:
                    creds = json.load(f)
                    self.username = creds.get('username')
                    self.password = creds.get('password')
                    self.shared_secret = creds.get('shared_secret')
                    self.identity_secret = creds.get('identity_secret')
            
            # Also check environment variables
            self.username = self.username or os.getenv('STEAM_USERNAME')
            self.password = self.password or os.getenv('STEAM_PASSWORD')
            self.shared_secret = self.shared_secret or os.getenv('STEAM_SHARED_SECRET')
            self.identity_secret = self.identity_secret or os.getenv('STEAM_IDENTITY_SECRET')
            
        except Exception as e:
            self.logger.error(f"Error loading Steam credentials: {e}")
    
    def save_credentials(self, username: str, password: str, shared_secret: str = None, identity_secret: str = None):
        """Save Steam credentials securely"""
        try:
            credentials = {
                'username': username,
                'password': password,
                'shared_secret': shared_secret,
                'identity_secret': identity_secret,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.credentials_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            # Set file permissions to be readable only by owner
            os.chmod(self.credentials_file, 0o600)
            
            self.username = username
            self.password = password
            self.shared_secret = shared_secret
            self.identity_secret = identity_secret
            
            self.logger.info("Steam credentials saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving Steam credentials: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """Check if we have a valid Steam session"""
        if not STEAMPY_AVAILABLE:
            return False
        
        try:
            return self.client and self.client.is_session_alive()
        except Exception as e:
            self.logger.error(f"Error checking authentication: {e}")
            return False
    
    def login(self) -> bool:
        """Login to Steam"""
        if not STEAMPY_AVAILABLE:
            self.logger.error("steampy library not available")
            return False
        
        if not self.username or not self.password:
            self.logger.error("Steam credentials not provided")
            return False
        
        try:
            self.logger.info(f"Attempting to login to Steam as {self.username}")
            
            # Create Steam client
            if self.shared_secret:
                self.client = SteamClient(
                    username=self.username,
                    password=self.password,
                    steam_guard=self.shared_secret
                )
            else:
                self.client = SteamClient(
                    username=self.username,
                    password=self.password
                )
            
            # Attempt login
            self.client.login()
            
            if self.is_authenticated():
                self.logger.info("Successfully logged into Steam")
                self._save_session()
                return True
            else:
                self.logger.error("Steam login failed - authentication check failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Steam login error: {e}")
            return False
    
    def logout(self):
        """Logout from Steam"""
        try:
            if self.client:
                self.client.logout()
                self.client = None
                self.logger.info("Logged out from Steam")
        except Exception as e:
            self.logger.error(f"Error during logout: {e}")
    
    def _save_session(self):
        """Save current session for reuse"""
        try:
            if self.client and hasattr(self.client, 'session'):
                session_data = {
                    'session_id': getattr(self.client.session, 'session_id', None),
                    'login_secure': getattr(self.client.session, 'login_secure', None),
                    'created_at': datetime.now().isoformat()
                }
                
                with open(self.session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                os.chmod(self.session_file, 0o600)
                
        except Exception as e:
            self.logger.error(f"Error saving session: {e}")
    
    def get_steam_id(self) -> Optional[str]:
        """Get the Steam ID of the authenticated account"""
        try:
            if self.client and hasattr(self.client, 'steam_id'):
                return str(self.client.steam_id)
        except Exception as e:
            self.logger.error(f"Error getting Steam ID: {e}")
        return None
    
    def get_inventory(self, app_id: int = 730, context_id: str = "2") -> Optional[Dict]:
        """Get inventory for specified app (730 = CS2/CSGO)"""
        try:
            if not self.is_authenticated():
                self.logger.error("Not authenticated to Steam")
                return None
            
            inventory = self.client.get_my_inventory(GameOptions.CS)
            return inventory
            
        except Exception as e:
            self.logger.error(f"Error getting inventory: {e}")
            return None
    
    def get_trade_offers(self) -> Optional[Dict]:
        """Get current trade offers"""
        try:
            if not self.is_authenticated():
                self.logger.error("Not authenticated to Steam")
                return None
            
            offers = self.client.get_trade_offers()
            return offers
            
        except Exception as e:
            self.logger.error(f"Error getting trade offers: {e}")
            return None

class SteamMarketTrader:
    """Enhanced Steam Market integration with trading capabilities"""
    
    def __init__(self, authenticator: SteamAuthenticator):
        self.auth = authenticator
        self.logger = logging.getLogger(__name__)
        self.config = FloatCheckerConfig()
    
    def get_item_price(self, market_hash_name: str, currency: Currency = Currency.USD) -> Optional[Dict]:
        """Get current market price for an item"""
        try:
            if not self.auth.is_authenticated():
                self.logger.error("Not authenticated to Steam")
                return None
            
            # Get price from Steam Market
            price_data = self.auth.client.market.fetch_price(
                market_hash_name, 
                GameOptions.CS,
                currency
            )
            
            return price_data
            
        except Exception as e:
            self.logger.error(f"Error getting item price for {market_hash_name}: {e}")
            return None
    
    def create_sell_listing(self, asset_id: str, price: int) -> bool:
        """Create a sell listing on Steam Market"""
        try:
            if not self.auth.is_authenticated():
                self.logger.error("Not authenticated to Steam")
                return False
            
            # Create sell order
            response = self.auth.client.market.create_sell_order(
                assetid=asset_id,
                game=GameOptions.CS,
                money_to_receive=price
            )
            
            if response.get('success'):
                self.logger.info(f"Successfully created sell listing for asset {asset_id} at ${price/100:.2f}")
                return True
            else:
                self.logger.error(f"Failed to create sell listing: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating sell listing: {e}")
            return False
    
    def get_market_listings(self) -> Optional[Dict]:
        """Get current market listings"""
        try:
            if not self.auth.is_authenticated():
                return None
            
            listings = self.auth.client.market.get_my_market_listings()
            return listings
            
        except Exception as e:
            self.logger.error(f"Error getting market listings: {e}")
            return None
    
    def cancel_listing(self, listing_id: str) -> bool:
        """Cancel a market listing"""
        try:
            if not self.auth.is_authenticated():
                return False
            
            response = self.auth.client.market.cancel_sell_order(listing_id)
            return response.get('success', False)
            
        except Exception as e:
            self.logger.error(f"Error canceling listing {listing_id}: {e}")
            return False

def create_steam_client() -> Optional[SteamAuthenticator]:
    """Factory function to create authenticated Steam client"""
    auth = SteamAuthenticator()
    
    if auth.login():
        return auth
    else:
        logging.error("Failed to create authenticated Steam client")
        return None

# Test function
def test_steam_connection():
    """Test Steam authentication and basic functionality"""
    print("🧪 Testing Steam Connection...")
    
    auth = SteamAuthenticator()
    
    if not STEAMPY_AVAILABLE:
        print("❌ steampy library not available")
        print("Install with: pip install steampy")
        return False
    
    if not auth.username:
        print("❌ Steam credentials not configured")
        print("Set STEAM_USERNAME and STEAM_PASSWORD environment variables")
        return False
    
    print(f"Attempting login for user: {auth.username}")
    
    if auth.login():
        print("✅ Steam authentication successful")
        
        steam_id = auth.get_steam_id()
        print(f"Steam ID: {steam_id}")
        
        # Test inventory access
        inventory = auth.get_inventory()
        if inventory:
            print(f"✅ Inventory access successful ({len(inventory)} items)")
        else:
            print("⚠️ Could not access inventory")
        
        auth.logout()
        return True
    else:
        print("❌ Steam authentication failed")
        return False

if __name__ == "__main__":
    test_steam_connection()