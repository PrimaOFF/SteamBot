# CS2 Float Checker

An automated tool to scan the Steam Market for CS2 skins with rare float values. This tool helps identify undervalued skins with exceptionally low or high float values that could be valuable for trading or investment.

## Features

- **Automated Market Scanning**: Continuously monitors Steam Market listings for CS2 skins
- **Float Value Analysis**: Identifies rare float values (very low FN or very high BS)
- **Skin-Specific Ranges**: Handles different float ranges for different skins
- **Rarity Scoring**: Assigns rarity scores (0-100) to help identify the most valuable finds
- **Database Storage**: Stores all findings in SQLite database for historical analysis
- **Real-time Notifications**: Alerts when rare items are found
- **Export Functionality**: Export findings to JSON for further analysis

## Installation

1. Install Python 3.7+ and pip
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Steam API key (optional but recommended):
```bash
export STEAM_API_KEY="your_steam_api_key_here"
```

## Usage

### Basic Scanning
Scan specific items:
```bash
python float_checker.py --items "AK-47 | Redline" "AWP | Dragon Lore"
```

### Continuous Monitoring
Run continuous scanning with 30-minute intervals:
```bash
python float_checker.py --continuous --interval 30
```

### Export Results
Export rare finds to JSON:
```bash
python float_checker.py --export rare_items.json
```

### View Statistics
Check database statistics:
```bash
python float_checker.py --stats
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

## Legal Notice

This tool is for educational and research purposes. Always comply with Steam's Terms of Service and API usage policies. Use rate limiting and respect server resources.

## Contributing

Contributions welcome! Please submit pull requests with:
- New skin-specific float ranges
- Improved rarity scoring algorithms
- Additional third-party API integrations
- Better market analysis features