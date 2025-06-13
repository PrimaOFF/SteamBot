#!/usr/bin/env python3

"""
Test script to verify Telegram setup improvements
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_bot_token_validation():
    """Test bot token validation"""
    from setup import FloatCheckerSetup
    
    setup = FloatCheckerSetup()
    
    print("Testing bot token validation:")
    
    # Valid tokens
    valid_tokens = [
        "123456789:ABCdefGHIjklMNOpqrsTUVwxyz-1234567890",
        "987654321:XYZabcDEFghiJKLmnoPQRstu_ABCDEFGHIJ",
    ]
    
    # Invalid tokens
    invalid_tokens = [
        "123456789",  # No colon
        "abc:123456789",  # First part not numbers
        "123:abc",  # Second part too short
        "",  # Empty
        "123:456:789",  # Multiple colons
    ]
    
    print("\n✅ Valid tokens:")
    for token in valid_tokens:
        result = setup._validate_bot_token(token)
        print(f"  {token[:20]}... → {result}")
    
    print("\n❌ Invalid tokens:")
    for token in invalid_tokens:
        result = setup._validate_bot_token(token)
        print(f"  '{token}' → {result}")

def test_telegram_error_handling():
    """Test Telegram error handling"""
    from telegram_bot import TelegramNotifier
    
    print("\n" + "="*50)
    print("Testing Telegram error handling:")
    
    # Test with no credentials
    print("\n1. No credentials:")
    notifier = TelegramNotifier()
    print(f"Enabled: {notifier.enabled}")
    
    # Test with invalid credentials
    print("\n2. Invalid credentials:")
    notifier = TelegramNotifier("invalid_token", "invalid_chat")
    result = notifier.send_message("Test message")
    print(f"Send result: {result}")

def main():
    print("🧪 Testing Telegram Setup Improvements")
    print("=" * 50)
    
    try:
        test_bot_token_validation()
        test_telegram_error_handling()
        
        print("\n✅ All tests completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all modules are available")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()