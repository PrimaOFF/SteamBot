#!/usr/bin/env python3

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import math

from config import FloatCheckerConfig
from database import FloatDatabase
from skin_database import SkinDatabase

class ScanPriority(Enum):
    CRITICAL = "critical"  # High-value items, frequent scanning
    HIGH = "high"         # Popular items, regular scanning  
    MEDIUM = "medium"     # Standard items, normal scanning
    LOW = "low"          # Low-value items, infrequent scanning
    SUSPENDED = "suspended"  # Temporarily disabled

@dataclass
class ScanTarget:
    """Represents an item target for scanning with intelligence"""
    item_name: str
    priority: ScanPriority
    base_value: float  # Estimated base market value
    last_scanned: Optional[datetime] = None
    scan_count: int = 0
    extreme_floats_found: int = 0
    total_value_found: float = 0.0
    success_rate: float = 0.0  # Percentage of scans that found extreme floats
    avg_scan_duration: float = 0.0  # Average time to scan this item
    cooldown_until: Optional[datetime] = None
    priority_boost: float = 1.0  # Multiplier for dynamic priority adjustment
    failure_count: int = 0  # Consecutive scan failures
    
    def __post_init__(self):
        if self.last_scanned is None:
            self.last_scanned = datetime.now() - timedelta(days=1)  # Default to 1 day ago
    
    @property
    def is_on_cooldown(self) -> bool:
        """Check if item is currently on cooldown"""
        return self.cooldown_until and datetime.now() < self.cooldown_until
    
    @property
    def time_since_last_scan(self) -> timedelta:
        """Get time since last scan"""
        return datetime.now() - self.last_scanned if self.last_scanned else timedelta(days=365)
    
    @property
    def dynamic_priority_score(self) -> float:
        """Calculate dynamic priority score for intelligent ordering"""
        base_scores = {
            ScanPriority.CRITICAL: 1000.0,
            ScanPriority.HIGH: 100.0,
            ScanPriority.MEDIUM: 10.0,
            ScanPriority.LOW: 1.0,
            ScanPriority.SUSPENDED: 0.0
        }
        
        score = base_scores[self.priority]
        
        # Apply success rate multiplier (higher success = higher priority)
        score *= (1.0 + self.success_rate)
        
        # Apply value multiplier (higher value = higher priority)
        score *= (1.0 + math.log10(max(self.base_value, 1.0)))
        
        # Apply time decay (longer since last scan = higher priority)
        hours_since_scan = self.time_since_last_scan.total_seconds() / 3600
        time_multiplier = min(hours_since_scan / 24, 5.0)  # Cap at 5x after 5 days
        score *= (1.0 + time_multiplier)
        
        # Apply priority boost
        score *= self.priority_boost
        
        # Reduce score for items with many failures
        if self.failure_count > 0:
            score *= (1.0 / (1.0 + self.failure_count * 0.2))
        
        return score

class SmartScanner:
    """Intelligent scanning system with prioritization and optimization"""
    
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.database = FloatDatabase()
        self.skin_db = SkinDatabase()
        self.logger = logging.getLogger(__name__)
        
        # Scan targets management
        self.scan_targets: Dict[str, ScanTarget] = {}
        self.scan_history = deque(maxlen=1000)  # Last 1000 scan results
        
        # Scanning intervals (in hours)
        self.scan_intervals = {
            ScanPriority.CRITICAL: 1.0,   # Every hour
            ScanPriority.HIGH: 4.0,       # Every 4 hours
            ScanPriority.MEDIUM: 12.0,    # Every 12 hours
            ScanPriority.LOW: 48.0,       # Every 2 days
            ScanPriority.SUSPENDED: 168.0  # Every week (effectively disabled)
        }
        
        # Performance tracking
        self.performance_stats = {
            'total_scans': 0,
            'successful_scans': 0,
            'extreme_floats_found': 0,
            'total_scan_time': 0.0,
            'avg_scan_time': 0.0,
            'last_optimization': datetime.now()
        }
        
        # Load existing data
        self._load_scan_targets()
        self._load_performance_stats()
    
    def _load_scan_targets(self):
        """Load scan targets from storage"""
        try:
            targets_file = "scan_targets.json"
            if os.path.exists(targets_file):
                with open(targets_file, 'r') as f:
                    data = json.load(f)
                    
                for target_data in data.get('targets', []):
                    target = ScanTarget(
                        item_name=target_data['item_name'],
                        priority=ScanPriority(target_data['priority']),
                        base_value=target_data.get('base_value', 10.0),
                        scan_count=target_data.get('scan_count', 0),
                        extreme_floats_found=target_data.get('extreme_floats_found', 0),
                        total_value_found=target_data.get('total_value_found', 0.0),
                        success_rate=target_data.get('success_rate', 0.0),
                        avg_scan_duration=target_data.get('avg_scan_duration', 0.0),
                        priority_boost=target_data.get('priority_boost', 1.0),
                        failure_count=target_data.get('failure_count', 0)
                    )
                    
                    # Parse datetime fields
                    if target_data.get('last_scanned'):
                        target.last_scanned = datetime.fromisoformat(target_data['last_scanned'])
                    if target_data.get('cooldown_until'):
                        target.cooldown_until = datetime.fromisoformat(target_data['cooldown_until'])
                    
                    self.scan_targets[target.item_name] = target
                
                self.logger.info(f"Loaded {len(self.scan_targets)} scan targets")
            else:
                # Initialize with default targets
                self._initialize_default_targets()
                
        except Exception as e:
            self.logger.error(f"Error loading scan targets: {e}")
            self._initialize_default_targets()
    
    def _initialize_default_targets(self):
        """Initialize with default high-value targets"""
        self.logger.info("Initializing default scan targets...")
        
        # Critical priority items (highest value)
        critical_items = [
            ("AWP | Dragon Lore", 8000.0),
            ("M4A4 | Howl", 3000.0),
            ("AK-47 | Fire Serpent", 2000.0),
            ("Karambit | Fade", 1500.0),
            ("Butterfly Knife | Fade", 1200.0)
        ]
        
        # High priority items
        high_priority_items = [
            ("AK-47 | Redline", 150.0),
            ("AWP | Asiimov", 200.0),
            ("Karambit | Doppler", 800.0),
            ("M4A1-S | Hot Rod", 300.0),
            ("Glock-18 | Fade", 400.0),
            ("USP-S | Kill Confirmed", 250.0)
        ]
        
        # Medium priority items
        medium_priority_items = [
            ("AK-47 | Vulcan", 80.0),
            ("AWP | Hyper Beast", 60.0),
            ("M4A4 | Dragon King", 40.0),
            ("Glock-18 | Water Elemental", 30.0),
            ("USP-S | Orion", 50.0)
        ]
        
        # Add critical items
        for item_name, base_value in critical_items:
            self.scan_targets[item_name] = ScanTarget(
                item_name=item_name,
                priority=ScanPriority.CRITICAL,
                base_value=base_value
            )
        
        # Add high priority items
        for item_name, base_value in high_priority_items:
            self.scan_targets[item_name] = ScanTarget(
                item_name=item_name,
                priority=ScanPriority.HIGH,
                base_value=base_value
            )
        
        # Add medium priority items
        for item_name, base_value in medium_priority_items:
            self.scan_targets[item_name] = ScanTarget(
                item_name=item_name,
                priority=ScanPriority.MEDIUM,
                base_value=base_value
            )
        
        self.logger.info(f"Initialized {len(self.scan_targets)} default scan targets")
        self._save_scan_targets()
    
    def _load_performance_stats(self):
        """Load performance statistics"""
        try:
            stats_file = "scanner_performance.json"
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    saved_stats = json.load(f)
                    self.performance_stats.update(saved_stats)
                    
                    # Parse datetime field
                    if 'last_optimization' in saved_stats:
                        self.performance_stats['last_optimization'] = datetime.fromisoformat(saved_stats['last_optimization'])
                
                self.logger.info("Performance statistics loaded")
        except Exception as e:
            self.logger.error(f"Error loading performance stats: {e}")
    
    def get_next_scan_targets(self, max_targets: int = 10) -> List[ScanTarget]:
        """Get the next items to scan based on intelligent prioritization"""
        try:
            # Filter available targets (not on cooldown, not suspended)
            available_targets = []
            
            for target in self.scan_targets.values():
                if target.priority == ScanPriority.SUSPENDED:
                    continue
                
                if target.is_on_cooldown:
                    continue
                
                # Check if enough time has passed since last scan
                interval_hours = self.scan_intervals[target.priority]
                time_since_scan = target.time_since_last_scan.total_seconds() / 3600
                
                if time_since_scan >= interval_hours:
                    available_targets.append(target)
            
            # Sort by dynamic priority score (highest first)
            available_targets.sort(key=lambda x: x.dynamic_priority_score, reverse=True)
            
            # Return top targets
            selected_targets = available_targets[:max_targets]
            
            self.logger.info(f"Selected {len(selected_targets)} targets for scanning")
            for i, target in enumerate(selected_targets[:5], 1):
                self.logger.info(f"  {i}. {target.item_name} (Priority: {target.priority.value}, Score: {target.dynamic_priority_score:.1f})")
            
            return selected_targets
            
        except Exception as e:
            self.logger.error(f"Error getting next scan targets: {e}")
            return []
    
    def record_scan_result(self, item_name: str, scan_duration: float, 
                          extreme_floats_found: int, total_value_found: float, 
                          success: bool = True):
        """Record the result of a scan for learning and optimization"""
        try:
            if item_name not in self.scan_targets:
                self.logger.warning(f"Recording result for unknown target: {item_name}")
                return
            
            target = self.scan_targets[item_name]
            
            # Update scan statistics
            target.scan_count += 1
            target.last_scanned = datetime.now()
            
            if success:
                target.extreme_floats_found += extreme_floats_found
                target.total_value_found += total_value_found
                target.failure_count = 0  # Reset failure count on success
            else:
                target.failure_count += 1
                # Set cooldown for failed items
                if target.failure_count >= 3:
                    target.cooldown_until = datetime.now() + timedelta(hours=6)
                    self.logger.warning(f"Setting 6-hour cooldown for {item_name} after {target.failure_count} failures")
            
            # Update success rate
            successful_scans = target.scan_count - target.failure_count
            target.success_rate = successful_scans / target.scan_count if target.scan_count > 0 else 0.0
            
            # Update average scan duration
            if target.avg_scan_duration == 0:
                target.avg_scan_duration = scan_duration
            else:
                target.avg_scan_duration = (target.avg_scan_duration + scan_duration) / 2
            
            # Update performance stats
            self.performance_stats['total_scans'] += 1
            if success:
                self.performance_stats['successful_scans'] += 1
                self.performance_stats['extreme_floats_found'] += extreme_floats_found
            
            self.performance_stats['total_scan_time'] += scan_duration
            self.performance_stats['avg_scan_time'] = (
                self.performance_stats['total_scan_time'] / self.performance_stats['total_scans']
            )
            
            # Add to scan history
            self.scan_history.append({
                'item_name': item_name,
                'timestamp': datetime.now().isoformat(),
                'duration': scan_duration,
                'extreme_floats_found': extreme_floats_found,
                'value_found': total_value_found,
                'success': success
            })
            
            self.logger.info(f"Recorded scan result for {item_name}: "
                           f"Duration: {scan_duration:.1f}s, "
                           f"Extreme floats: {extreme_floats_found}, "
                           f"Value: ${total_value_found:.2f}")
            
            # Auto-save after each result
            self._save_scan_targets()
            self._save_performance_stats()
            
        except Exception as e:
            self.logger.error(f"Error recording scan result: {e}")
    
    def optimize_priorities(self):
        """Optimize priorities based on historical performance"""
        try:
            self.logger.info("Optimizing scan priorities based on performance data...")
            
            optimization_count = 0
            
            for target in self.scan_targets.values():
                if target.scan_count < 3:  # Need at least 3 scans for optimization
                    continue
                
                old_priority = target.priority
                
                # Calculate performance metrics
                value_per_scan = target.total_value_found / target.scan_count if target.scan_count > 0 else 0
                efficiency_score = value_per_scan / max(target.avg_scan_duration, 1.0)
                
                # Adjust priority based on performance
                if efficiency_score > 50.0 and target.success_rate > 0.2:
                    # High performing item - boost priority
                    if target.priority == ScanPriority.MEDIUM:
                        target.priority = ScanPriority.HIGH
                        optimization_count += 1
                    elif target.priority == ScanPriority.LOW:
                        target.priority = ScanPriority.MEDIUM
                        optimization_count += 1
                        
                elif efficiency_score < 5.0 or target.success_rate < 0.05:
                    # Poor performing item - reduce priority
                    if target.priority == ScanPriority.HIGH:
                        target.priority = ScanPriority.MEDIUM
                        optimization_count += 1
                    elif target.priority == ScanPriority.MEDIUM:
                        target.priority = ScanPriority.LOW
                        optimization_count += 1
                    elif target.priority == ScanPriority.LOW and target.success_rate == 0:
                        target.priority = ScanPriority.SUSPENDED
                        optimization_count += 1
                
                # Apply dynamic priority boost for consistently good performers
                if target.success_rate > 0.3 and efficiency_score > 20.0:
                    target.priority_boost = min(target.priority_boost * 1.1, 2.0)
                elif target.success_rate < 0.1:
                    target.priority_boost = max(target.priority_boost * 0.9, 0.5)
                
                if target.priority != old_priority:
                    self.logger.info(f"Priority changed for {target.item_name}: "
                                   f"{old_priority.value} → {target.priority.value} "
                                   f"(Success: {target.success_rate:.1%}, "
                                   f"Efficiency: {efficiency_score:.1f})")
            
            self.performance_stats['last_optimization'] = datetime.now()
            
            self.logger.info(f"Priority optimization completed: {optimization_count} items adjusted")
            
            # Save changes
            self._save_scan_targets()
            self._save_performance_stats()
            
        except Exception as e:
            self.logger.error(f"Error optimizing priorities: {e}")
    
    def add_scan_target(self, item_name: str, priority: ScanPriority, base_value: float):
        """Add a new scan target"""
        try:
            if item_name in self.scan_targets:
                self.logger.warning(f"Scan target already exists: {item_name}")
                return False
            
            self.scan_targets[item_name] = ScanTarget(
                item_name=item_name,
                priority=priority,
                base_value=base_value
            )
            
            self.logger.info(f"Added scan target: {item_name} (Priority: {priority.value}, Value: ${base_value})")
            self._save_scan_targets()
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding scan target: {e}")
            return False
    
    def remove_scan_target(self, item_name: str):
        """Remove a scan target"""
        try:
            if item_name not in self.scan_targets:
                self.logger.warning(f"Scan target not found: {item_name}")
                return False
            
            del self.scan_targets[item_name]
            self.logger.info(f"Removed scan target: {item_name}")
            self._save_scan_targets()
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing scan target: {e}")
            return False
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary and statistics"""
        try:
            # Calculate additional metrics
            success_rate = (self.performance_stats['successful_scans'] / 
                          max(self.performance_stats['total_scans'], 1))
            
            # Get top performers
            top_performers = sorted(
                [t for t in self.scan_targets.values() if t.scan_count > 0],
                key=lambda x: x.total_value_found,
                reverse=True
            )[:5]
            
            # Get priority distribution
            priority_distribution = defaultdict(int)
            for target in self.scan_targets.values():
                priority_distribution[target.priority.value] += 1
            
            return {
                'total_targets': len(self.scan_targets),
                'total_scans': self.performance_stats['total_scans'],
                'success_rate': success_rate,
                'extreme_floats_found': self.performance_stats['extreme_floats_found'],
                'avg_scan_time': self.performance_stats['avg_scan_time'],
                'last_optimization': self.performance_stats['last_optimization'].isoformat(),
                'priority_distribution': dict(priority_distribution),
                'top_performers': [
                    {
                        'item_name': t.item_name,
                        'total_value_found': t.total_value_found,
                        'success_rate': t.success_rate,
                        'scan_count': t.scan_count
                    }
                    for t in top_performers
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def _save_scan_targets(self):
        """Save scan targets to storage"""
        try:
            targets_data = {
                'targets': [],
                'last_updated': datetime.now().isoformat()
            }
            
            for target in self.scan_targets.values():
                target_data = {
                    'item_name': target.item_name,
                    'priority': target.priority.value,
                    'base_value': target.base_value,
                    'scan_count': target.scan_count,
                    'extreme_floats_found': target.extreme_floats_found,
                    'total_value_found': target.total_value_found,
                    'success_rate': target.success_rate,
                    'avg_scan_duration': target.avg_scan_duration,
                    'priority_boost': target.priority_boost,
                    'failure_count': target.failure_count
                }
                
                if target.last_scanned:
                    target_data['last_scanned'] = target.last_scanned.isoformat()
                if target.cooldown_until:
                    target_data['cooldown_until'] = target.cooldown_until.isoformat()
                
                targets_data['targets'].append(target_data)
            
            with open("scan_targets.json", 'w') as f:
                json.dump(targets_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving scan targets: {e}")
    
    def _save_performance_stats(self):
        """Save performance statistics"""
        try:
            stats_to_save = self.performance_stats.copy()
            
            # Convert datetime to string
            if 'last_optimization' in stats_to_save:
                stats_to_save['last_optimization'] = stats_to_save['last_optimization'].isoformat()
            
            with open("scanner_performance.json", 'w') as f:
                json.dump(stats_to_save, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving performance stats: {e}")

# Test function
def test_smart_scanner():
    """Test smart scanner functionality"""
    print("🧪 Testing Smart Scanner...")
    
    try:
        scanner = SmartScanner()
        
        # Test 1: Get next scan targets
        print("Test 1: Getting next scan targets")
        targets = scanner.get_next_scan_targets(5)
        print(f"✅ Got {len(targets)} scan targets")
        
        for i, target in enumerate(targets[:3], 1):
            print(f"  {i}. {target.item_name} - Priority: {target.priority.value} - Score: {target.dynamic_priority_score:.1f}")
        
        # Test 2: Record scan results
        print("\nTest 2: Recording scan results")
        if targets:
            test_target = targets[0]
            scanner.record_scan_result(
                item_name=test_target.item_name,
                scan_duration=45.0,
                extreme_floats_found=2,
                total_value_found=150.0,
                success=True
            )
            print(f"✅ Recorded scan result for {test_target.item_name}")
        
        # Test 3: Performance summary
        print("\nTest 3: Performance summary")
        summary = scanner.get_performance_summary()
        print("✅ Performance Summary:")
        for key, value in summary.items():
            if key != 'top_performers':
                print(f"  {key}: {value}")
        
        # Test 4: Add new target
        print("\nTest 4: Adding new scan target")
        success = scanner.add_scan_target(
            "Test Item | Test Skin",
            ScanPriority.MEDIUM,
            100.0
        )
        print(f"✅ Add target result: {success}")
        
        print("\n✅ Smart Scanner test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Smart Scanner test failed: {e}")
        return False

if __name__ == "__main__":
    import os
    test_smart_scanner()