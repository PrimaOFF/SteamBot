#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path

def load_environment():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def main():
    print("🎯 CS2 Float Checker")
    print("=" * 50)
    
    # Load environment variables
    load_environment()
    
    # Check if configured
    if not os.path.exists(".env") or not os.path.exists("user_config.json"):
        print("❌ Not configured yet!")
        print("Please run: python3 setup.py")
        return
    
    # Load user configuration
    try:
        with open("user_config.json") as f:
            user_config = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    # Check Steam API key
    if not os.getenv('STEAM_API_KEY'):
        print("❌ Steam API key not found!")
        print("Please run: python3 setup.py --reconfigure")
        return
    
    print("✅ Configuration loaded")
    print(f"📊 Min rarity score: {user_config.get('MIN_RARITY_SCORE', 70)}")
    print(f"⏱️ Scan interval: {user_config.get('SCAN_INTERVAL', 30)} minutes")
    print(f"📈 Max listings per item: {user_config.get('MAX_LISTINGS', 50)}")
    
    # Check Telegram configuration
    telegram_configured = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
    print(f"📱 Telegram notifications: {'✅ Enabled' if telegram_configured else '❌ Disabled'}")
    
    print("\n🚀 Starting Float Checker...")
    
    # Import and run the main checker
    try:
        from float_checker import CS2FloatChecker
        
        checker = CS2FloatChecker()
        
        # Test Telegram if configured
        if telegram_configured:
            print("📱 Testing Telegram connection...")
            if checker.test_telegram():
                print("✅ Telegram connection successful")
            else:
                print("⚠️ Telegram connection failed")
        
        print("\n🔍 Choose scanning mode:")
        print("1. 🚀 ENHANCED: Full market scan (optimized)")
        print("2. ⚡ ENHANCED: Continuous aggressive scanning")
        print("3. 📊 Standard: Scan monitored items only")
        print("4. 🔄 Standard: Continuous scanning (slow)")
        print("5. 🎯 Custom item scan")
        print("6. 📈 Show statistics")
        print("7. 🧪 Test API performance")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            print("🚀 Starting ENHANCED full market scan...")
            print("⚡ This will scan the ENTIRE CS2 market with maximum efficiency!")
            confirm = input("Continue? (y/n): ").lower().startswith('y')
            if confirm:
                import asyncio
                from enhanced_float_checker import EnhancedFloatChecker
                enhanced_checker = EnhancedFloatChecker()
                asyncio.run(enhanced_checker.scan_entire_market_optimized())
            
        elif choice == "2":
            print("⚡ Starting ENHANCED continuous aggressive scanning...")
            interval = input(f"Scan interval in minutes (default: 5, minimum: 1): ").strip()
            try:
                interval = max(1, int(interval)) if interval else 5
            except ValueError:
                interval = 5
            
            print(f"🔄 Starting aggressive scanning every {interval} minutes...")
            import asyncio
            from enhanced_float_checker import EnhancedFloatChecker
            enhanced_checker = EnhancedFloatChecker()
            asyncio.run(enhanced_checker.continuous_aggressive_scan(interval))
            
        elif choice == "3":
            print("📊 Scanning monitored items (standard mode)...")
            checker.scan_multiple_items(checker.config.MONITORED_ITEMS, user_config.get('MAX_LISTINGS', 50))
            
        elif choice == "4":
            print(f"🔄 Starting standard continuous scanning (every {user_config.get('SCAN_INTERVAL', 30)} minutes)...")
            checker.continuous_scan(checker.config.MONITORED_ITEMS, user_config.get('SCAN_INTERVAL', 30))
            
        elif choice == "5":
            items = input("Enter item names (separated by commas): ").strip()
            if items:
                item_list = [item.strip() for item in items.split(',')]
                checker.scan_multiple_items(item_list, user_config.get('MAX_LISTINGS', 50))
            else:
                print("❌ No items specified")
                
        elif choice == "6":
            stats = checker.database.get_statistics()
            print("\n📊 Database Statistics:")
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
                
        elif choice == "7":
            print("🧪 Testing API performance...")
            import asyncio
            from enhanced_float_checker import EnhancedFloatChecker
            enhanced_checker = EnhancedFloatChecker()
            # Run performance test
            import subprocess
            result = subprocess.run([
                sys.executable, "enhanced_float_checker.py", "--test-performance"
            ], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
        else:
            print("❌ Invalid choice")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\n👋 Stopping Float Checker...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()