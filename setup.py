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
        print("-" * 30)
        
        config = {}
        
        # Steam API Key
        print("\n🔑 Steam API Key Setup")
        print("Get your Steam API key from: https://steamcommunity.com/dev/apikey")
        
        while True:
            steam_api_key = input("Enter your Steam API key: ").strip()
            if len(steam_api_key) == 32:  # Steam API keys are 32 characters
                config['STEAM_API_KEY'] = steam_api_key
                break
            else:
                print("❌ Invalid Steam API key format. Please try again.")
        
        # Telegram Configuration
        print("\n📱 Telegram Bot Setup")
        print("1. Create a bot: Message @BotFather on Telegram")
        print("2. Send /newbot and follow instructions")
        print("3. Get your bot token")
        
        setup_telegram = input("Do you want to setup Telegram notifications? (y/n): ").lower().startswith('y')
        
        if setup_telegram:
            while True:
                bot_token = input("Enter your Telegram bot token: ").strip()
                if bot_token and ':' in bot_token:
                    config['TELEGRAM_BOT_TOKEN'] = bot_token
                    break
                else:
                    print("❌ Invalid bot token format. Please try again.")
            
            print("\n📋 Getting your Chat ID:")
            print("1. Message your bot on Telegram")
            print("2. Visit: https://api.telegram.org/bot{}/getUpdates".format(bot_token))
            print("3. Look for 'chat':{'id': YOUR_CHAT_ID}")
            
            while True:
                chat_id = input("Enter your Telegram chat ID: ").strip()
                if chat_id.lstrip('-').isdigit():  # Can be negative for groups
                    config['TELEGRAM_CHAT_ID'] = chat_id
                    break
                else:
                    print("❌ Invalid chat ID format. Please try again.")
        
        # Scanning preferences
        print("\n🎯 Scanning Preferences")
        
        min_rarity_score = input("Minimum rarity score for notifications (default: 70): ").strip()
        if min_rarity_score and min_rarity_score.isdigit():
            config['MIN_RARITY_SCORE'] = int(min_rarity_score)
        else:
            config['MIN_RARITY_SCORE'] = 70
        
        scan_interval = input("Continuous scan interval in minutes (default: 30): ").strip()
        if scan_interval and scan_interval.isdigit():
            config['SCAN_INTERVAL'] = int(scan_interval)
        else:
            config['SCAN_INTERVAL'] = 30
        
        max_listings = input("Max listings to check per item (default: 50): ").strip()
        if max_listings and max_listings.isdigit():
            config['MAX_LISTINGS'] = int(max_listings)
        else:
            config['MAX_LISTINGS'] = 50
        
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
            # Test Steam API
            print("Testing Steam API...")
            
            # Import and test
            from steam_market_api import SteamMarketAPI
            api = SteamMarketAPI()
            
            # Simple test - search for a common item
            result = api.search_items("AK-47", count=1)
            if result:
                print("✅ Steam API test passed")
            else:
                print("⚠️ Steam API test inconclusive")
            
            # Test Telegram if configured
            if os.path.exists(self.env_file):
                with open(self.env_file) as f:
                    env_content = f.read()
                    if 'TELEGRAM_BOT_TOKEN' in env_content:
                        print("Testing Telegram bot...")
                        
                        from telegram_bot import TelegramNotifier
                        telegram = TelegramNotifier()
                        
                        if telegram.test_connection():
                            print("✅ Telegram test passed")
                        else:
                            print("⚠️ Telegram test failed - check your credentials")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            return False
    
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