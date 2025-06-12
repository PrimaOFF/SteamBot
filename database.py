import sqlite3
import json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import asdict
from float_analyzer import FloatAnalysis
from config import FloatCheckerConfig

class FloatDatabase:
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.db_path = self.config.DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS float_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_name TEXT NOT NULL,
                    wear_condition TEXT NOT NULL,
                    float_value REAL NOT NULL,
                    price REAL NOT NULL,
                    rarity_score REAL NOT NULL,
                    is_rare BOOLEAN NOT NULL,
                    analysis_timestamp TEXT NOT NULL,
                    inspect_link TEXT,
                    market_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS skin_ranges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skin_name TEXT UNIQUE NOT NULL,
                    ranges_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_name TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER,
                    timestamp TEXT NOT NULL,
                    source TEXT DEFAULT 'steam_market'
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_item_name ON float_analyses(item_name)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_rarity_score ON float_analyses(rarity_score)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON float_analyses(analysis_timestamp)
            ''')
    
    def save_analysis(self, analysis: FloatAnalysis) -> int:
        """Save a float analysis to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO float_analyses 
                (item_name, wear_condition, float_value, price, rarity_score, 
                 is_rare, analysis_timestamp, inspect_link, market_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.item_name,
                analysis.wear_condition,
                analysis.float_value,
                analysis.price,
                analysis.rarity_score,
                analysis.is_rare,
                analysis.analysis_timestamp.isoformat(),
                analysis.inspect_link,
                analysis.market_url
            ))
            return cursor.lastrowid
    
    def get_rare_items(self, min_rarity_score: float = 70.0, limit: int = 100) -> List[Dict]:
        """Get rare items from the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM float_analyses 
                WHERE rarity_score >= ? 
                ORDER BY rarity_score DESC, analysis_timestamp DESC
                LIMIT ?
            ''', (min_rarity_score, limit))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_item_history(self, item_name: str, days: int = 30) -> List[Dict]:
        """Get historical data for a specific item"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM float_analyses 
                WHERE item_name = ? 
                AND datetime(analysis_timestamp) >= datetime('now', '-{} days')
                ORDER BY analysis_timestamp DESC
            '''.format(days), (item_name,))
            return [dict(row) for row in cursor.fetchall()]
    
    def save_skin_ranges(self, skin_name: str, ranges: Dict):
        """Save custom float ranges for a specific skin"""
        ranges_json = json.dumps(ranges)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO skin_ranges (skin_name, ranges_json)
                VALUES (?, ?)
            ''', (skin_name, ranges_json))
    
    def get_skin_ranges(self, skin_name: str) -> Optional[Dict]:
        """Get custom float ranges for a specific skin"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT ranges_json FROM skin_ranges WHERE skin_name = ?
            ''', (skin_name,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None
    
    def save_market_data(self, item_name: str, price: float, volume: int = 0, source: str = 'steam_market'):
        """Save market data point"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO market_data (item_name, price, volume, timestamp, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (item_name, price, volume, datetime.now().isoformat(), source))
    
    def get_price_trends(self, item_name: str, days: int = 7) -> List[Dict]:
        """Get price trends for an item"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM market_data 
                WHERE item_name = ? 
                AND datetime(timestamp) >= datetime('now', '-{} days')
                ORDER BY timestamp ASC
            '''.format(days), (item_name,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_top_items_by_rarity(self, limit: int = 50) -> List[Dict]:
        """Get top items by rarity score"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT item_name, MAX(rarity_score) as max_rarity, 
                       AVG(price) as avg_price, COUNT(*) as count
                FROM float_analyses 
                GROUP BY item_name 
                ORDER BY max_rarity DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data to keep database size manageable"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                DELETE FROM float_analyses 
                WHERE datetime(analysis_timestamp) < datetime('now', '-{} days')
            '''.format(days))
            
            conn.execute('''
                DELETE FROM market_data 
                WHERE datetime(timestamp) < datetime('now', '-{} days')
            '''.format(days))
            
            conn.execute('VACUUM')
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Total analyses
            cursor = conn.execute('SELECT COUNT(*) FROM float_analyses')
            stats['total_analyses'] = cursor.fetchone()[0]
            
            # Rare items count
            cursor = conn.execute('SELECT COUNT(*) FROM float_analyses WHERE is_rare = 1')
            stats['rare_items_count'] = cursor.fetchone()[0]
            
            # Unique items tracked
            cursor = conn.execute('SELECT COUNT(DISTINCT item_name) FROM float_analyses')
            stats['unique_items'] = cursor.fetchone()[0]
            
            # Average rarity score
            cursor = conn.execute('SELECT AVG(rarity_score) FROM float_analyses')
            avg_rarity = cursor.fetchone()[0]
            stats['average_rarity_score'] = round(avg_rarity, 2) if avg_rarity else 0
            
            return stats