#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from pathlib import Path

class FloatCheckerSetup:
    def __init__(self):
        self.config_file = "user_config.json"
        self.env_file = ".env"
        
    def main(self):
        print("🎯 CS2 Float Checker Setup")
        print("=" * 50)
        
        # Check Python version
        if sys.version_info < (3, 7):
            print("❌ Python 3.7 or higher is required")
            return False
        
        print("✅ Python version check passed")
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Setup configuration
        if not self.setup_configuration():
            return False
        
        # Test configuration
        if not self.test_configuration():
            return False
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run: python3 float_checker.py --test-telegram")
        print("2. Run: python3 float_checker.py --all-weapons")
        print("3. For continuous monitoring: python3 float_checker.py --continuous")
        
        return True
    
    def install_dependencies(self):
        """Install required Python packages"""
        print("\n📦 Installing dependencies...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please install manually: pip install -r requirements.txt")
            return False
        except FileNotFoundError:
            print("❌ requirements.txt not found")
            return False
    
    def setup_configuration(self):
        """Setup user configuration"""
        print("\n⚙️ Configuration Setup")
        print("-" * 50)
        
        config = {}
        
        # Steam API Key
        print("\n🔑 Steam API Key Setup")
        print("Purpose: Required to access Steam Market data and prices")
        print("Get your key from: https://steamcommunity.com/dev/apikey")
        print("Note: You need a Steam account with a purchase history")
        
        while True:
            steam_api_key = input("\nEnter your Steam API key (32 characters): ").strip()
            if len(steam_api_key) == 32 and steam_api_key.isalnum():
                config['STEAM_API_KEY'] = steam_api_key
                print("✅ Steam API key validated")
                break
            else:
                print("❌ Invalid format. Steam API keys are exactly 32 alphanumeric characters")
        
        # Telegram Configuration
        print("\n📱 Telegram Bot Setup")
        print("Purpose: Get instant notifications when rare float items are found")
        print("Benefits: Real-time alerts, daily summaries, error notifications")
        print("\nSetup steps:")
        print("1. Open Telegram and message @BotFather")
        print("2. Send /newbot command")
        print("3. Follow instructions to create your bot")
        print("4. Copy the bot token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)")
        
        setup_telegram = input("\nDo you want to setup Telegram notifications? (y/n): ").lower().startswith('y')
        
        if setup_telegram:
            # Bot token setup with validation
            while True:
                print("\nEnter your Telegram bot token:")
                bot_token = input("Token: ").strip()
                
                if self._validate_bot_token(bot_token):
                    config['TELEGRAM_BOT_TOKEN'] = bot_token
                    print("✅ Bot token format validated")
                    break
                else:
                    print("❌ Invalid bot token format")
                    print("Expected format: numbers:letters (e.g., 123456789:ABCdefGHI...)")
            
            # Auto-detect chat ID
            print("\n📋 Chat ID Detection")
            print("Next steps:")
            print("1. Send a message to your bot on Telegram (any message)")
            print("2. Press Enter here after sending the message")
            
            input("Press Enter after messaging your bot...")
            
            chat_id = self._detect_chat_id(bot_token)
            if chat_id:
                config['TELEGRAM_CHAT_ID'] = chat_id
                print(f"✅ Auto-detected Chat ID: {chat_id}")
            else:
                print("❌ Auto-detection failed. Please enter manually:")
                print("To find your chat ID:")
                print(f"1. Visit: https://api.telegram.org/bot{bot_token}/getUpdates")
                print("2. Look for 'chat':{'id': YOUR_CHAT_ID}")
                
                while True:
                    chat_id = input("Enter your Telegram chat ID: ").strip()
                    if chat_id.lstrip('-').isdigit():
                        config['TELEGRAM_CHAT_ID'] = chat_id
                        print("✅ Chat ID saved")
                        break
                    else:
                        print("❌ Invalid chat ID format (should be a number)")
            
            # Test Telegram connection immediately
            if self._test_telegram_connection(bot_token, config['TELEGRAM_CHAT_ID']):
                print("✅ Telegram test successful!")
            else:
                print("⚠️ Telegram test failed - but configuration saved anyway")
        
        # Scanning preferences with detailed explanations
        print("\n🎯 Scanning Preferences")
        print("-" * 30)
        
        print("\n📊 Minimum Rarity Score (0-100):")
        print("• 70+: Only very rare floats (recommended for notifications)")
        print("• 50+: Moderately rare floats (more notifications)")
        print("• 30+: All somewhat interesting floats (many notifications)")
        print("Higher scores = fewer but more valuable notifications")
        
        min_rarity_score = input("Minimum rarity score for notifications (default: 70): ").strip()
        if min_rarity_score and min_rarity_score.isdigit() and 0 <= int(min_rarity_score) <= 100:
            config['MIN_RARITY_SCORE'] = int(min_rarity_score)
        else:
            config['MIN_RARITY_SCORE'] = 70
        print(f"✅ Set to {config['MIN_RARITY_SCORE']}")
        
        print("\n⏱️ Continuous Scan Interval:")
        print("• 15 min: High frequency scanning (more API calls)")
        print("• 30 min: Balanced scanning (recommended)")
        print("• 60 min: Low frequency scanning (less API calls)")
        print("Lower intervals = faster detection but more resource usage")
        
        scan_interval = input("Continuous scan interval in minutes (default: 30): ").strip()
        if scan_interval and scan_interval.isdigit() and int(scan_interval) >= 5:
            config['SCAN_INTERVAL'] = int(scan_interval)
        else:
            config['SCAN_INTERVAL'] = 30
        print(f"✅ Set to {config['SCAN_INTERVAL']} minutes")
        
        print("\n📈 Max Listings Per Item:")
        print("• 25: Quick scanning (may miss some items)")
        print("• 50: Balanced scanning (recommended)")
        print("• 100: Thorough scanning (slower but more complete)")
        print("Higher values = more thorough but slower scanning")
        
        max_listings = input("Max listings to check per item (default: 50): ").strip()
        if max_listings and max_listings.isdigit() and int(max_listings) >= 10:
            config['MAX_LISTINGS'] = int(max_listings)
        else:
            config['MAX_LISTINGS'] = 50
        print(f"✅ Set to {config['MAX_LISTINGS']} listings")
        
        # Save configuration
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Also create .env file for environment variables
            with open(self.env_file, 'w') as f:
                f.write(f"STEAM_API_KEY={config['STEAM_API_KEY']}\n")
                if 'TELEGRAM_BOT_TOKEN' in config:
                    f.write(f"TELEGRAM_BOT_TOKEN={config['TELEGRAM_BOT_TOKEN']}\n")
                    f.write(f"TELEGRAM_CHAT_ID={config['TELEGRAM_CHAT_ID']}\n")
            
            print(f"✅ Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False
    
    def test_configuration(self):
        """Test the configuration"""
        print("\n🧪 Testing Configuration")
        print("-" * 30)
        
        try:
            # Load environment variables first
            self._load_environment_variables()
            
            # Test Steam API
            print("Testing Steam API connection...")
            
            steam_api_key = os.getenv('STEAM_API_KEY')
            if not steam_api_key:
                print("❌ Steam API key not found in environment")
                return False
            
            # Import and test
            from steam_market_api import SteamMarketAPI
            api = SteamMarketAPI()
            
            # Simple test - search for a common item
            result = api.search_items("AK-47", count=1)
            if result and result.get('results'):
                print("✅ Steam API test passed")
            else:
                print("⚠️ Steam API test inconclusive - may be rate limited")
            
            # Test Telegram if configured
            telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
            telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if telegram_token and telegram_chat_id:
                print("Testing Telegram bot connection...")
                
                from telegram_bot import TelegramNotifier
                telegram = TelegramNotifier()
                
                if telegram.test_connection():
                    print("✅ Telegram test passed")
                else:
                    print("⚠️ Telegram test failed - check your credentials")
            else:
                print("ℹ️ Telegram not configured - skipping test")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            print(f"Error details: {str(e)}")
            return False
    
    def _validate_bot_token(self, token: str) -> bool:
        """Validate Telegram bot token format"""
        if not token or ':' not in token:
            return False
        
        parts = token.split(':')
        if len(parts) != 2:
            return False
        
        # First part should be numbers (bot ID)
        if not parts[0].isdigit():
            return False
        
        # Second part should be letters/numbers (token)
        if len(parts[1]) < 35 or not parts[1].replace('_', '').replace('-', '').isalnum():
            return False
        
        return True
    
    def _detect_chat_id(self, bot_token: str) -> str:
        """Auto-detect chat ID from bot updates"""
        try:
            import requests
            url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('ok') and data.get('result'):
                # Get the most recent message
                for update in reversed(data['result']):
                    if 'message' in update and 'chat' in update['message']:
                        chat_id = str(update['message']['chat']['id'])
                        return chat_id
            
        except Exception as e:
            print(f"Chat ID detection error: {e}")
        
        return None
    
    def _test_telegram_connection(self, bot_token: str, chat_id: str) -> bool:
        """Test Telegram connection"""
        try:
            import requests
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            test_message = """
🧪 <b>Setup Test</b>

CS2 Float Checker setup completed successfully!

<b>✅ Bot Token:</b> Valid
<b>✅ Chat ID:</b> Detected  
<b>✅ Connection:</b> Working

You're all set to receive rare float notifications! 🎯
"""
            
            payload = {
                'chat_id': chat_id,
                'text': test_message.strip(),
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get('ok', False)
            
        except Exception as e:
            print(f"Telegram test error: {e}")
            return False
    
    def _load_environment_variables(self):
        """Load environment variables from .env file"""
        if os.path.exists(self.env_file):
            with open(self.env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
    
    def load_existing_config(self):
        """Load existing configuration if available"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except:
                pass
        return {}

def main():
    setup = FloatCheckerSetup()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--reconfigure':
        print("🔄 Reconfiguring CS2 Float Checker...")
    else:
        # Check if already configured
        if os.path.exists("user_config.json") and os.path.exists(".env"):
            print("⚠️ Configuration files already exist.")
            reconfigure = input("Do you want to reconfigure? (y/n): ").lower().startswith('y')
            if not reconfigure:
                print("Setup cancelled. Use --reconfigure to force reconfiguration.")
                return
    
    success = setup.main()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()