# CS2 Float Checker

An advanced automated tool that scans the Steam Market for CS2 skins with rare float values. This tool helps identify undervalued skins with exceptionally low or high float values that could be valuable for trading or investment.

## 🚀 Quick Start

### Simple Setup (Recommended)
1. Clone this repository
2. Run the setup script:
```bash
python3 setup.py
```
3. Follow the **enhanced interactive setup**:
   - Steam API key configuration with validation
   - **Automatic Telegram bot setup** with chat ID detection
   - Clear explanations of all configuration options
   - **Instant connection testing** for both Steam and Telegram
4. Start scanning:
```bash
python3 run.py
```

### 🔧 Enhanced Setup Features
- **Auto-detection** of Telegram chat ID (no manual entry needed!)
- **Smart validation** of bot tokens and API keys
- **Detailed explanations** of what each setting does
- **Immediate testing** - know if it works right away
- **Better error messages** with specific solutions

### Manual Setup
1. Install Python 3.7+ and pip
2. Install dependencies: `pip install -r requirements.txt`
3. Get Steam API key from: https://steamcommunity.com/dev/apikey
4. Set environment variables:
```bash
export STEAM_API_KEY="your_steam_api_key_here"
export TELEGRAM_BOT_TOKEN="your_bot_token"  # Optional
export TELEGRAM_CHAT_ID="your_chat_id"     # Optional
```

## ✨ Features

- **🔍 Comprehensive Scanning**: Scans ALL CS2 weapons and skins from database
- **🎯 Smart Float Analysis**: Uses actual skin-specific float ranges (not just 0-1)
- **📱 Telegram Notifications**: Real-time alerts for rare float finds
- **📊 Advanced Rarity Scoring**: Dynamic scoring based on actual float distributions
- **💾 Database Tracking**: SQLite database for historical analysis
- **🔄 Continuous Monitoring**: Automated scanning with configurable intervals
- **📈 Export Functionality**: JSON export for further analysis
- **🛡️ Rate Limiting**: Respects Steam's API limits

## 📋 Usage

### Interactive Mode (Easiest)
```bash
python3 run.py
```
Choose from scanning options:
1. Scan all weapons (comprehensive)
2. Scan monitored items only  
3. Continuous scanning
4. Custom item scan
5. Show statistics

### Command Line Interface
```bash
# Scan all weapons from database
python3 float_checker.py --all-weapons

# Continuous monitoring
python3 float_checker.py --continuous --interval 30

# Scan specific items
python3 float_checker.py --items "AK-47 | Redline" "AWP | Dragon Lore"

# Test Telegram notifications
python3 float_checker.py --test-telegram

# Export rare finds
python3 float_checker.py --export rare_finds.json

# Show database statistics
python3 float_checker.py --stats
```

## Configuration

Edit `config.py` to customize:

- **MONITORED_ITEMS**: List of items to scan by default
- **RARE_FLOAT_THRESHOLDS**: Define what constitutes a rare float
- **SKIN_SPECIFIC_RANGES**: Add custom float ranges for specific skins
- **REQUEST_DELAY**: Adjust rate limiting between API calls

## Float Value Ranges

### Standard Wear Conditions:
- **Factory New**: 0.00 - 0.07
- **Minimal Wear**: 0.07 - 0.15  
- **Field-Tested**: 0.15 - 0.37
- **Well-Worn**: 0.37 - 0.45
- **Battle-Scarred**: 0.45 - 1.00

### Rare Float Thresholds:
- **Factory New**: < 0.005 (Very rare), < 0.001 (Extremely rare)
- **Battle-Scarred**: > 0.995 (Very rare), > 0.999 (Extremely rare)

## Rarity Scoring

The tool assigns rarity scores (0-100) based on:
- Float value position within wear range
- Historical rarity of similar floats
- Market demand for specific float ranges

### Score Ranges:
- **90-100**: Extremely rare, strong investment potential
- **70-89**: Very rare, good investment potential  
- **50-69**: Moderately rare, watch for opportunities
- **30-49**: Common, standard market value
- **0-29**: Very common, avoid for investment

## Database Schema

The tool stores data in SQLite with tables for:
- **float_analyses**: Individual item analyses
- **skin_ranges**: Custom float ranges for specific skins
- **market_data**: Historical price and volume data

## Third-Party Integration

The tool is designed to work with third-party float APIs:
- **CSFloat API**: For accurate float value extraction
- **Tradeit.gg API**: Alternative float checking service

Note: You'll need to implement the actual API calls to these services for production use.

## Limitations

- Steam Market API has rate limits
- Float values require third-party APIs for accurate extraction
- Some skins have restricted float ranges not covered by standard ranges
- Market prices can fluctuate rapidly

## 🔧 Troubleshooting

### Telegram Setup Issues

**Problem: "Chat ID detection failed"**
- Solution: Make sure you sent a message to your bot first
- Alternative: Get chat ID manually from `https://api.telegram.org/bot<TOKEN>/getUpdates`

**Problem: "Invalid bot token format"**
- Solution: Token should look like `123456789:ABCdefGHI...` (numbers:letters)
- Get a new token from @BotFather on Telegram

**Problem: "Telegram test failed"**
- Check internet connection
- Verify bot token is correct
- Make sure you started the bot with @BotFather

### Steam API Issues

**Problem: "Invalid Steam API key"**
- Key must be exactly 32 characters
- Get from: https://steamcommunity.com/dev/apikey
- Requires Steam account with purchase history

**Problem: "Steam API test inconclusive"**
- Often due to rate limiting (normal)
- Wait a few minutes and try again
- Key is likely valid if format is correct

### General Issues

**Problem: Dependencies won't install**
- Try: `pip install requests urllib3`
- On Windows: Use `python -m pip install -r requirements.txt`
- On Linux: May need `python3-pip` package

**Problem: Script won't run**
- Ensure Python 3.7+ is installed
- Check file permissions: `chmod +x setup.py run.py`
- Try running with full path: `python3 /full/path/to/setup.py`

## Legal Notice

This tool is for educational and research purposes. Always comply with Steam's Terms of Service and API usage policies. Use rate limiting and respect server resources.

## Contributing

Contributions welcome! Please submit pull requests with:
- New skin-specific float ranges
- Improved rarity scoring algorithms
- Additional third-party API integrations
- Better market analysis features