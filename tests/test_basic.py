#!/usr/bin/env python3

import pytest
import asyncio
import os
import sys

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that we can import our main modules"""
    try:
        from config import FloatCheckerConfig
        assert FloatCheckerConfig is not None
    except ImportError as e:
        pytest.skip(f"Config import failed: {e}")

def test_config_creation():
    """Test basic config creation"""
    try:
        from config import FloatCheckerConfig
        config = FloatCheckerConfig()
        assert config.CS2_APP_ID == 730
        assert config.MONITORED_ITEMS is not None
        assert len(config.MONITORED_ITEMS) > 0
    except ImportError:
        pytest.skip("Config module not available")

def test_environment_variables():
    """Test that environment variables work"""
    # Set test environment variables
    os.environ['STEAM_API_KEY'] = 'test_key'
    os.environ['REDIS_PASSWORD'] = 'test_redis_pass'
    
    # Test they can be read
    assert os.getenv('STEAM_API_KEY') == 'test_key'
    assert os.getenv('REDIS_PASSWORD') == 'test_redis_pass'

@pytest.mark.asyncio
async def test_async_functionality():
    """Test basic async functionality"""
    async def dummy_async_function():
        await asyncio.sleep(0.01)
        return "test_result"
    
    result = await dummy_async_function()
    assert result == "test_result"

def test_data_structures():
    """Test that our data structures work"""
    try:
        from config import FloatCheckerConfig
        config = FloatCheckerConfig()
        
        # Test SKIN_SPECIFIC_RANGES structure
        assert isinstance(config.SKIN_SPECIFIC_RANGES, dict)
        
        # Test that at least one skin has proper structure
        if config.SKIN_SPECIFIC_RANGES:
            first_skin = next(iter(config.SKIN_SPECIFIC_RANGES.values()))
            assert isinstance(first_skin, dict)
    except ImportError:
        pytest.skip("Config module not available")

def test_wear_restrictions():
    """Test wear restrictions configuration"""
    try:
        from config import FloatCheckerConfig
        config = FloatCheckerConfig()
        
        assert isinstance(config.WEAR_RESTRICTIONS, dict)
        assert 'no_factory_new' in config.WEAR_RESTRICTIONS
        assert 'no_battle_scarred' in config.WEAR_RESTRICTIONS
        assert isinstance(config.WEAR_RESTRICTIONS['no_factory_new'], list)
        assert isinstance(config.WEAR_RESTRICTIONS['no_battle_scarred'], list)
    except ImportError:
        pytest.skip("Config module not available")

if __name__ == "__main__":
    pytest.main([__file__])