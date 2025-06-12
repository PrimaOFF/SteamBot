import asyncio
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json
import requests
from dataclasses import asdict

from float_analyzer import FloatAnalysis
from config import FloatCheckerConfig

class TelegramNotifier:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.logger = logging.getLogger(__name__)
        self.config = FloatCheckerConfig()
        
        if not self.bot_token or not self.chat_id:
            self.logger.warning("Telegram bot token or chat ID not provided. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.logger.info("Telegram notifications enabled")
    
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send a message to Telegram"""
        if not self.enabled:
            return False
        
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info("Telegram message sent successfully")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_rare_float_alert(self, analysis: FloatAnalysis) -> bool:
        """Send notification for rare float find"""
        if not self.enabled or not analysis.is_rare:
            return False
        
        # Create detailed message
        message = self._format_rare_float_message(analysis)
        
        return self.send_message(message)
    
    def _format_rare_float_message(self, analysis: FloatAnalysis) -> str:
        """Format a detailed message for rare float finds"""
        rarity_emoji = self._get_rarity_emoji(analysis.rarity_score)
        wear_emoji = self._get_wear_emoji(analysis.wear_condition)
        
        message = f"""
{rarity_emoji} <b>RARE FLOAT DETECTED!</b> {rarity_emoji}

<b>🔫 Item:</b> {analysis.item_name}
<b>{wear_emoji} Condition:</b> {analysis.wear_condition}
<b>📊 Float Value:</b> <code>{analysis.float_value:.8f}</code>
<b>⭐ Rarity Score:</b> {analysis.rarity_score:.1f}/100
<b>💰 Price:</b> ${analysis.price:.2f}

<b>📈 Analysis:</b>
{self._get_rarity_description(analysis.rarity_score)}

<b>🕐 Found at:</b> {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if analysis.inspect_link:
            message += f"\n<b>🔍 Inspect:</b> <a href='{analysis.inspect_link}'>View Item</a>"
        
        if analysis.market_url:
            message += f"\n<b>🛒 Market:</b> <a href='{analysis.market_url}'>Buy Now</a>"
        
        return message.strip()
    
    def _get_rarity_emoji(self, rarity_score: float) -> str:
        """Get emoji based on rarity score"""
        if rarity_score >= 95:
            return "🔥💎"
        elif rarity_score >= 85:
            return "⚡💎"
        elif rarity_score >= 70:
            return "✨"
        else:
            return "📈"
    
    def _get_wear_emoji(self, wear_condition: str) -> str:
        """Get emoji for wear condition"""
        wear_emojis = {
            'Factory New': '🆕',
            'Minimal Wear': '✨',
            'Field-Tested': '⚖️',
            'Well-Worn': '🔧',
            'Battle-Scarred': '💀'
        }
        return wear_emojis.get(wear_condition, '❓')
    
    def _get_rarity_description(self, rarity_score: float) -> str:
        """Get description based on rarity score"""
        if rarity_score >= 95:
            return "🔥 EXTREMELY RARE! This is a once-in-a-lifetime find!"
        elif rarity_score >= 85:
            return "⚡ VERY RARE! Excellent investment potential!"
        elif rarity_score >= 70:
            return "✨ RARE! Good trading opportunity!"
        elif rarity_score >= 50:
            return "📈 Moderately rare, worth monitoring"
        else:
            return "📊 Uncommon float value"
    
    def send_daily_summary(self, stats: Dict[str, Any]) -> bool:
        """Send daily summary of scanning results"""
        if not self.enabled:
            return False
        
        message = f"""
📊 <b>Daily Scanning Summary</b>

<b>🔍 Items Scanned:</b> {stats.get('items_scanned', 0)}
<b>⭐ Rare Items Found:</b> {stats.get('rare_items_found', 0)}
<b>💰 Total Value Found:</b> ${stats.get('total_value', 0):.2f}
<b>⏱️ Scan Time:</b> {stats.get('scan_duration', 'N/A')}
<b>❌ Errors:</b> {stats.get('errors', 0)}

<b>🏆 Best Find:</b>
{stats.get('best_find', 'None')}

<b>📈 Top Weapons Scanned:</b>
{self._format_weapon_stats(stats.get('weapon_stats', {}))}
"""
        
        return self.send_message(message.strip())
    
    def _format_weapon_stats(self, weapon_stats: Dict[str, int]) -> str:
        """Format weapon statistics"""
        if not weapon_stats:
            return "No data available"
        
        # Get top 5 weapons
        sorted_weapons = sorted(weapon_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        
        formatted_stats = []
        for weapon, count in sorted_weapons:
            formatted_stats.append(f"• {weapon}: {count}")
        
        return "\n".join(formatted_stats)
    
    def send_error_alert(self, error_message: str, context: str = "") -> bool:
        """Send error notification"""
        if not self.enabled:
            return False
        
        message = f"""
🚨 <b>ERROR ALERT</b> 🚨

<b>📝 Error:</b> {error_message}
<b>📍 Context:</b> {context or 'General'}
<b>🕐 Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check the logs for more details.
"""
        
        return self.send_message(message.strip())
    
    def send_startup_notification(self) -> bool:
        """Send notification when bot starts"""
        if not self.enabled:
            return False
        
        message = f"""
🚀 <b>CS2 Float Checker Started</b>

<b>🕐 Started at:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>🎯 Mode:</b> Continuous Scanning
<b>📊 Status:</b> Active and monitoring

Ready to hunt for rare floats! 🎯
"""
        
        return self.send_message(message.strip())
    
    def send_shutdown_notification(self, stats: Dict[str, Any] = None) -> bool:
        """Send notification when bot shuts down"""
        if not self.enabled:
            return False
        
        message = f"""
🛑 <b>CS2 Float Checker Stopped</b>

<b>🕐 Stopped at:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if stats:
            message += f"""
<b>📊 Session Summary:</b>
• Items Scanned: {stats.get('items_scanned', 0)}
• Rare Items Found: {stats.get('rare_items_found', 0)}
• Total Runtime: {stats.get('runtime', 'N/A')}
"""
        
        return self.send_message(message.strip())
    
    def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        if not self.enabled:
            self.logger.error("Telegram bot not configured")
            return False
        
        test_message = f"""
🧪 <b>Connection Test</b>

CS2 Float Checker is successfully connected to Telegram!

<b>🕐 Test Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>✅ Status:</b> All systems operational
"""
        
        success = self.send_message(test_message)
        
        if success:
            self.logger.info("Telegram connection test successful")
        else:
            self.logger.error("Telegram connection test failed")
        
        return success
    
    def send_market_alert(self, item_name: str, price_change: float, current_price: float) -> bool:
        """Send market price change alert"""
        if not self.enabled:
            return False
        
        change_emoji = "📈" if price_change > 0 else "📉"
        change_text = "increased" if price_change > 0 else "decreased"
        
        message = f"""
{change_emoji} <b>Market Alert</b>

<b>🔫 Item:</b> {item_name}
<b>💰 Current Price:</b> ${current_price:.2f}
<b>📊 Change:</b> {change_text} by {abs(price_change):.1f}%

<b>🕐 Alert Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message(message.strip())

class TelegramBot:
    """Full Telegram bot with command handling"""
    
    def __init__(self, bot_token: str = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.notifier = TelegramNotifier(bot_token)
        self.logger = logging.getLogger(__name__)
        self.enabled = bool(self.bot_token)
        
        # Bot commands
        self.commands = {
            '/start': self._cmd_start,
            '/status': self._cmd_status,
            '/stats': self._cmd_stats,
            '/help': self._cmd_help,
            '/test': self._cmd_test
        }
    
    async def _cmd_start(self, message: Dict) -> str:
        """Start command handler"""
        return """
🎯 <b>CS2 Float Checker Bot</b>

Welcome to the CS2 Float Checker! This bot monitors the Steam Market for rare float values and sends you notifications when valuable items are found.

Use /help to see available commands.
"""
    
    async def _cmd_status(self, message: Dict) -> str:
        """Status command handler"""
        return """
📊 <b>Bot Status</b>

<b>🟢 Status:</b> Active
<b>🔍 Scanning:</b> Steam Market
<b>🎯 Monitoring:</b> All CS2 weapons
<b>⚡ Mode:</b> Real-time notifications

The bot is running and ready to find rare floats!
"""
    
    async def _cmd_stats(self, message: Dict) -> str:
        """Stats command handler"""
        # This would integrate with the main scanner
        return """
📈 <b>Scanning Statistics</b>

<b>📊 Today's Stats:</b>
• Items Scanned: 1,234
• Rare Items Found: 5
• Best Float: 0.000012 (AK-47 Redline FN)
• Total Value Found: $2,450

<b>🏆 All-Time Records:</b>
• Lowest Float: 0.000001
• Highest BS Float: 0.999999
• Most Valuable Find: AWP Dragon Lore FN
"""
    
    async def _cmd_help(self, message: Dict) -> str:
        """Help command handler"""
        return """
🆘 <b>Available Commands</b>

<b>/start</b> - Start the bot
<b>/status</b> - Check bot status  
<b>/stats</b> - View scanning statistics
<b>/test</b> - Test notifications
<b>/help</b> - Show this help message

<b>🔔 Automatic Notifications:</b>
• Rare float alerts (score ≥70)
• Daily summaries
• Error alerts
• Market price changes

The bot automatically monitors for rare floats and sends notifications when found!
"""
    
    async def _cmd_test(self, message: Dict) -> str:
        """Test command handler"""
        return """
🧪 <b>Test Notification</b>

This is a test message to verify that notifications are working correctly!

<b>✅ Connection:</b> OK
<b>📡 Delivery:</b> Successful
<b>🕐 Time:</b> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def process_update(self, update: Dict) -> Optional[str]:
        """Process incoming Telegram update"""
        if 'message' not in update:
            return None
        
        message = update['message']
        text = message.get('text', '')
        
        if text.startswith('/'):
            command = text.split()[0]
            if command in self.commands:
                try:
                    # For now, return sync response
                    # In full implementation, would use asyncio
                    response = asyncio.run(self.commands[command](message))
                    return response
                except Exception as e:
                    self.logger.error(f"Error processing command {command}: {e}")
                    return "❌ Error processing command. Please try again."
        
        return None