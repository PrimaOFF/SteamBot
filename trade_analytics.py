#!/usr/bin/env python3

import asyncio
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

from trading_system import TradeRecord, PortfolioItem, TradeOpportunity, TradingSystem
from database import FloatDatabase

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for trading"""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    win_rate: float = 0.0
    average_profit_per_trade: float = 0.0
    average_loss_per_trade: float = 0.0
    profit_factor: float = 0.0  # Total profit / Total loss
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    portfolio_value: float = 0.0
    roi_percentage: float = 0.0
    trades_per_day: float = 0.0
    best_trade_profit: float = 0.0
    worst_trade_loss: float = 0.0
    average_hold_time: float = 0.0  # in hours
    risk_adjusted_return: float = 0.0

@dataclass
class MarketAnalysis:
    """Market trend and pattern analysis"""
    trending_items: List[str]
    declining_items: List[str]
    high_volume_items: List[str]
    profitable_categories: Dict[str, float]
    market_sentiment: str  # "bullish", "bearish", "neutral"
    volatility_index: float
    liquidity_score: float
    opportunity_score: float

class TradeAnalytics:
    """Advanced analytics and reporting for trading performance"""
    
    def __init__(self, trading_system: TradingSystem):
        self.trading_system = trading_system
        self.database = FloatDatabase()
        self.logger = logging.getLogger(__name__)
        
        # Analytics data
        self.performance_history = deque(maxlen=1000)  # Last 1000 performance snapshots
        self.trade_analysis_cache = {}
        self.market_data = defaultdict(list)
        
        # Setup output directories
        self.reports_dir = Path("reports")
        self.charts_dir = Path("charts")
        self.reports_dir.mkdir(exist_ok=True)
        self.charts_dir.mkdir(exist_ok=True)
    
    def calculate_performance_metrics(self, trade_history: List[TradeRecord], 
                                    portfolio: List[PortfolioItem]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        metrics = PerformanceMetrics()
        
        if not trade_history:
            return metrics
        
        # Basic trade statistics
        metrics.total_trades = len(trade_history)
        successful_trades = [t for t in trade_history if t.actual_profit and t.actual_profit > 0]
        failed_trades = [t for t in trade_history if t.actual_profit and t.actual_profit <= 0]
        
        metrics.successful_trades = len(successful_trades)
        metrics.failed_trades = len(failed_trades)
        metrics.win_rate = (metrics.successful_trades / metrics.total_trades) * 100 if metrics.total_trades > 0 else 0
        
        # Profit/Loss calculations
        profits = [t.actual_profit for t in successful_trades if t.actual_profit]
        losses = [abs(t.actual_profit) for t in failed_trades if t.actual_profit]
        
        metrics.total_profit = sum(profits) if profits else 0.0
        metrics.total_loss = sum(losses) if losses else 0.0
        metrics.net_profit = metrics.total_profit - metrics.total_loss
        
        # Average metrics
        metrics.average_profit_per_trade = metrics.total_profit / max(len(profits), 1)
        metrics.average_loss_per_trade = metrics.total_loss / max(len(losses), 1)
        
        # Risk metrics
        metrics.profit_factor = metrics.total_profit / max(metrics.total_loss, 1)
        
        # Portfolio metrics
        portfolio_value = sum(item.current_market_price for item in portfolio)
        initial_investment = sum(item.purchase_price for item in portfolio)
        metrics.portfolio_value = portfolio_value
        metrics.roi_percentage = ((portfolio_value - initial_investment) / max(initial_investment, 1)) * 100
        
        # Consecutive win/loss streaks
        self._calculate_streaks(trade_history, metrics)
        
        # Time-based metrics
        self._calculate_time_metrics(trade_history, metrics)
        
        # Advanced risk metrics
        self._calculate_risk_metrics(trade_history, metrics)
        
        return metrics
    
    def _calculate_streaks(self, trade_history: List[TradeRecord], metrics: PerformanceMetrics):
        """Calculate consecutive win/loss streaks"""
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in reversed(trade_history):  # Start from most recent
            if trade.actual_profit and trade.actual_profit > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.actual_profit and trade.actual_profit <= 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        metrics.consecutive_wins = current_wins
        metrics.consecutive_losses = current_losses
        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses
    
    def _calculate_time_metrics(self, trade_history: List[TradeRecord], metrics: PerformanceMetrics):
        """Calculate time-based performance metrics"""
        if not trade_history:
            return
        
        # Trades per day
        oldest_trade = min(trade_history, key=lambda t: t.created_at)
        days_trading = (datetime.now() - oldest_trade.created_at).days
        metrics.trades_per_day = len(trade_history) / max(days_trading, 1)
        
        # Average hold time
        completed_trades = [t for t in trade_history if t.completed_at]
        if completed_trades:
            hold_times = [(t.completed_at - t.created_at).total_seconds() / 3600 
                         for t in completed_trades]
            metrics.average_hold_time = np.mean(hold_times)
        
        # Best and worst trades
        profits = [t.actual_profit for t in trade_history if t.actual_profit]
        if profits:
            metrics.best_trade_profit = max(profits)
            metrics.worst_trade_loss = min(profits)
    
    def _calculate_risk_metrics(self, trade_history: List[TradeRecord], metrics: PerformanceMetrics):
        """Calculate advanced risk metrics"""
        if len(trade_history) < 2:
            return
        
        # Calculate returns for Sharpe ratio and drawdown
        returns = []
        cumulative_returns = []
        cumulative_profit = 0
        
        for trade in trade_history:
            if trade.actual_profit:
                returns.append(trade.actual_profit)
                cumulative_profit += trade.actual_profit
                cumulative_returns.append(cumulative_profit)
        
        if returns:
            # Sharpe ratio (simplified)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            metrics.sharpe_ratio = mean_return / max(std_return, 0.01)
            
            # Maximum drawdown
            if cumulative_returns:
                peak = cumulative_returns[0]
                max_dd = 0
                current_dd = 0
                
                for value in cumulative_returns:
                    if value > peak:
                        peak = value
                        current_dd = 0
                    else:
                        current_dd = peak - value
                        max_dd = max(max_dd, current_dd)
                
                metrics.max_drawdown = max_dd
                metrics.current_drawdown = cumulative_returns[-1] - peak if peak > cumulative_returns[-1] else 0
            
            # Risk-adjusted return
            metrics.risk_adjusted_return = mean_return / max(std_return, 0.01)
    
    def analyze_market_trends(self, trade_history: List[TradeRecord]) -> MarketAnalysis:
        """Analyze market trends and patterns"""
        try:
            # Item performance analysis
            item_performance = defaultdict(list)
            category_performance = defaultdict(list)
            
            for trade in trade_history:
                if trade.actual_profit:
                    item_name = trade.opportunity.item_name
                    item_performance[item_name].append(trade.actual_profit)
                    
                    # Extract weapon category
                    category = item_name.split(' | ')[0] if ' | ' in item_name else item_name.split(' ')[0]
                    category_performance[category].append(trade.actual_profit)
            
            # Calculate trending items
            trending_items = []
            declining_items = []
            
            for item, profits in item_performance.items():
                if len(profits) >= 3:  # Need at least 3 trades
                    avg_profit = np.mean(profits)
                    trend = np.polyfit(range(len(profits)), profits, 1)[0]  # Linear trend
                    
                    if trend > 0 and avg_profit > 0:
                        trending_items.append(item)
                    elif trend < 0 or avg_profit < 0:
                        declining_items.append(item)
            
            # High volume items (most traded)
            high_volume_items = sorted(item_performance.keys(), 
                                     key=lambda x: len(item_performance[x]), 
                                     reverse=True)[:10]
            
            # Profitable categories
            profitable_categories = {}
            for category, profits in category_performance.items():
                profitable_categories[category] = np.mean(profits) if profits else 0
            
            # Market sentiment
            recent_trades = trade_history[-20:] if len(trade_history) >= 20 else trade_history
            recent_profits = [t.actual_profit for t in recent_trades if t.actual_profit]
            
            if recent_profits:
                avg_recent_profit = np.mean(recent_profits)
                if avg_recent_profit > 5:
                    sentiment = "bullish"
                elif avg_recent_profit < -5:
                    sentiment = "bearish"
                else:
                    sentiment = "neutral"
            else:
                sentiment = "neutral"
            
            # Volatility and opportunity scores (simplified)
            volatility_index = np.std(recent_profits) if recent_profits else 0
            liquidity_score = len(high_volume_items) / 10.0  # Normalized
            opportunity_score = len(trending_items) / max(len(item_performance), 1)
            
            return MarketAnalysis(
                trending_items=trending_items[:10],
                declining_items=declining_items[:10],
                high_volume_items=high_volume_items,
                profitable_categories=dict(sorted(profitable_categories.items(), 
                                                key=lambda x: x[1], reverse=True)[:10]),
                market_sentiment=sentiment,
                volatility_index=volatility_index,
                liquidity_score=liquidity_score,
                opportunity_score=opportunity_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing market trends: {e}")
            return MarketAnalysis([], [], [], {}, "neutral", 0, 0, 0)
    
    def generate_performance_charts(self, trade_history: List[TradeRecord], 
                                  portfolio: List[PortfolioItem]) -> Dict[str, str]:
        """Generate comprehensive performance charts"""
        chart_paths = {}
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Profit/Loss over time
            chart_paths['profit_timeline'] = self._create_profit_timeline_chart(trade_history)
            
            # 2. Portfolio composition pie chart
            chart_paths['portfolio_composition'] = self._create_portfolio_composition_chart(portfolio)
            
            # 3. Win rate and trade frequency
            chart_paths['trading_stats'] = self._create_trading_stats_chart(trade_history)
            
            # 4. Item category performance
            chart_paths['category_performance'] = self._create_category_performance_chart(trade_history)
            
            # 5. Risk metrics dashboard
            chart_paths['risk_dashboard'] = self._create_risk_dashboard_chart(trade_history)
            
            self.logger.info(f"✅ Generated {len(chart_paths)} performance charts")
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
        
        return chart_paths
    
    def _create_profit_timeline_chart(self, trade_history: List[TradeRecord]) -> str:
        """Create profit timeline chart"""
        if not trade_history:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Cumulative profit
        dates = [t.completed_at or t.created_at for t in trade_history]
        profits = [t.actual_profit or 0 for t in trade_history]
        cumulative_profits = np.cumsum(profits)
        
        ax1.plot(dates, cumulative_profits, linewidth=2, marker='o', markersize=4)
        ax1.set_title('Cumulative Profit Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Profit ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Individual trade profits
        colors = ['green' if p > 0 else 'red' for p in profits]
        ax2.bar(range(len(profits)), profits, color=colors, alpha=0.7)
        ax2.set_title('Individual Trade Profits', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Profit ($)')
        ax2.set_xlabel('Trade Number')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        chart_path = self.charts_dir / "profit_timeline.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_portfolio_composition_chart(self, portfolio: List[PortfolioItem]) -> str:
        """Create portfolio composition pie chart"""
        if not portfolio:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # By value
        categories = defaultdict(float)
        for item in portfolio:
            category = item.item_name.split(' | ')[0] if ' | ' in item.item_name else item.item_name.split(' ')[0]
            categories[category] += item.current_market_price
        
        if categories:
            ax1.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%', startangle=90)
            ax1.set_title('Portfolio Composition by Value', fontsize=14, fontweight='bold')
        
        # By P&L
        pnl_data = defaultdict(float)
        for item in portfolio:
            category = item.item_name.split(' | ')[0] if ' | ' in item.item_name else item.item_name.split(' ')[0]
            pnl_data[category] += item.unrealized_profit
        
        if pnl_data:
            colors = ['green' if v > 0 else 'red' for v in pnl_data.values()]
            ax2.bar(pnl_data.keys(), pnl_data.values(), color=colors, alpha=0.7)
            ax2.set_title('Unrealized P&L by Category', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Unrealized P&L ($)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.charts_dir / "portfolio_composition.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_trading_stats_chart(self, trade_history: List[TradeRecord]) -> str:
        """Create trading statistics chart"""
        if not trade_history:
            return ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Win rate
        successful_trades = len([t for t in trade_history if t.actual_profit and t.actual_profit > 0])
        win_rate = (successful_trades / len(trade_history)) * 100
        
        ax1.pie([win_rate, 100 - win_rate], labels=['Wins', 'Losses'], 
                colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Win Rate: {win_rate:.1f}%', fontsize=14, fontweight='bold')
        
        # Trade frequency over time
        trade_dates = [t.created_at.date() for t in trade_history]
        date_counts = pd.Series(trade_dates).value_counts().sort_index()
        
        ax2.plot(date_counts.index, date_counts.values, marker='o')
        ax2.set_title('Daily Trade Frequency', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Trades')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Profit distribution
        profits = [t.actual_profit for t in trade_history if t.actual_profit]
        if profits:
            ax3.hist(profits, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax3.set_title('Profit Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Profit ($)')
            ax3.set_ylabel('Frequency')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.grid(True, alpha=0.3)
        
        # Trade size distribution
        trade_sizes = [t.opportunity.market_price for t in trade_history]
        if trade_sizes:
            ax4.hist(trade_sizes, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_title('Trade Size Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Trade Size ($)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.charts_dir / "trading_stats.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_category_performance_chart(self, trade_history: List[TradeRecord]) -> str:
        """Create category performance chart"""
        if not trade_history:
            return ""
        
        # Group by weapon category
        category_performance = defaultdict(list)
        for trade in trade_history:
            if trade.actual_profit:
                category = trade.opportunity.item_name.split(' | ')[0] if ' | ' in trade.opportunity.item_name else trade.opportunity.item_name.split(' ')[0]
                category_performance[category].append(trade.actual_profit)
        
        if not category_performance:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average profit by category
        avg_profits = {cat: np.mean(profits) for cat, profits in category_performance.items()}
        categories = list(avg_profits.keys())
        profits = list(avg_profits.values())
        colors = ['green' if p > 0 else 'red' for p in profits]
        
        ax1.bar(categories, profits, color=colors, alpha=0.7)
        ax1.set_title('Average Profit by Weapon Category', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Profit ($)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Trade volume by category
        trade_counts = {cat: len(profits) for cat, profits in category_performance.items()}
        ax2.bar(trade_counts.keys(), trade_counts.values(), color='blue', alpha=0.7)
        ax2.set_title('Trade Volume by Category', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Trades')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = self.charts_dir / "category_performance.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _create_risk_dashboard_chart(self, trade_history: List[TradeRecord]) -> str:
        """Create risk metrics dashboard"""
        if not trade_history:
            return ""
        
        metrics = self.calculate_performance_metrics(trade_history, [])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Drawdown chart
        profits = [t.actual_profit for t in trade_history if t.actual_profit]
        if profits:
            cumulative = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            
            ax1.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            ax1.plot(drawdown, color='red', linewidth=2)
            ax1.set_title('Drawdown Chart', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Drawdown ($)')
            ax1.set_xlabel('Trade Number')
            ax1.grid(True, alpha=0.3)
        
        # Risk metrics summary
        risk_metrics = [
            ('Sharpe Ratio', metrics.sharpe_ratio),
            ('Profit Factor', metrics.profit_factor),
            ('Max Drawdown', metrics.max_drawdown),
            ('Win Rate', metrics.win_rate)
        ]
        
        metric_names = [m[0] for m in risk_metrics]
        metric_values = [m[1] for m in risk_metrics]
        
        ax2.barh(metric_names, metric_values, color='skyblue', alpha=0.7)
        ax2.set_title('Risk Metrics Summary', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # Consecutive wins/losses
        streak_data = [
            ('Current Wins', metrics.consecutive_wins),
            ('Current Losses', metrics.consecutive_losses),
            ('Max Wins', metrics.max_consecutive_wins),
            ('Max Losses', metrics.max_consecutive_losses)
        ]
        
        streak_names = [s[0] for s in streak_data]
        streak_values = [s[1] for s in streak_data]
        colors = ['green', 'red', 'darkgreen', 'darkred']
        
        ax3.bar(streak_names, streak_values, color=colors, alpha=0.7)
        ax3.set_title('Win/Loss Streaks', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Trades')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Performance over time (monthly)
        if len(trade_history) > 30:
            monthly_profits = defaultdict(float)
            for trade in trade_history:
                if trade.actual_profit:
                    month_key = trade.created_at.strftime('%Y-%m')
                    monthly_profits[month_key] += trade.actual_profit
            
            months = list(monthly_profits.keys())
            profits = list(monthly_profits.values())
            colors = ['green' if p > 0 else 'red' for p in profits]
            
            ax4.bar(months, profits, color=colors, alpha=0.7)
            ax4.set_title('Monthly Performance', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Monthly Profit ($)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        chart_path = self.charts_dir / "risk_dashboard.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def generate_comprehensive_report(self, trade_history: List[TradeRecord], 
                                    portfolio: List[PortfolioItem]) -> str:
        """Generate comprehensive trading performance report"""
        try:
            # Calculate metrics
            metrics = self.calculate_performance_metrics(trade_history, portfolio)
            market_analysis = self.analyze_market_trends(trade_history)
            
            # Generate charts
            chart_paths = self.generate_performance_charts(trade_history, portfolio)
            
            # Create HTML report
            report_html = self._create_html_report(metrics, market_analysis, chart_paths, 
                                                 trade_history, portfolio)
            
            # Save report
            report_path = self.reports_dir / f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            self.logger.info(f"✅ Comprehensive report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return ""
    
    def _create_html_report(self, metrics: PerformanceMetrics, market_analysis: MarketAnalysis,
                           chart_paths: Dict[str, str], trade_history: List[TradeRecord],
                           portfolio: List[PortfolioItem]) -> str:
        """Create HTML report with all analytics"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CS2 Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                .chart-container {{ margin: 20px 0; text-align: center; }}
                .chart-container img {{ max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .section-title {{ color: #2c3e50; font-size: 1.5em; margin: 30px 0 15px 0; border-left: 4px solid #3498db; padding-left: 10px; }}
                .trade-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .trade-table th, .trade-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .trade-table th {{ background-color: #3498db; color: white; }}
                .trade-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .portfolio-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .portfolio-item {{ background-color: #ecf0f1; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
                .list-container {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 CS2 Trading Performance Report</h1>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <div class="section-title">📊 Key Performance Metrics</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value {'positive' if metrics.net_profit > 0 else 'negative'}">${metrics.net_profit:.2f}</div>
                        <div class="metric-label">Net Profit</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.win_rate:.1f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.total_trades}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {'positive' if metrics.roi_percentage > 0 else 'negative'}">{metrics.roi_percentage:.1f}%</div>
                        <div class="metric-label">ROI</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics.portfolio_value:.2f}</div>
                        <div class="metric-label">Portfolio Value</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.profit_factor:.2f}</div>
                        <div class="metric-label">Profit Factor</div>
                    </div>
                </div>
                
                <div class="section-title">📈 Performance Charts</div>
                {self._generate_chart_html(chart_paths)}
                
                <div class="section-title">🔍 Market Analysis</div>
                <div class="portfolio-grid">
                    <div class="portfolio-item">
                        <h4>📈 Trending Items</h4>
                        <div class="list-container">
                            {', '.join(market_analysis.trending_items[:5]) if market_analysis.trending_items else 'No trending items'}
                        </div>
                    </div>
                    <div class="portfolio-item">
                        <h4>📉 Declining Items</h4>
                        <div class="list-container">
                            {', '.join(market_analysis.declining_items[:5]) if market_analysis.declining_items else 'No declining items'}
                        </div>
                    </div>
                    <div class="portfolio-item">
                        <h4>🎯 Market Sentiment</h4>
                        <div class="list-container">
                            <strong>{market_analysis.market_sentiment.upper()}</strong>
                            <br>Volatility: {market_analysis.volatility_index:.2f}
                            <br>Opportunity Score: {market_analysis.opportunity_score:.2f}
                        </div>
                    </div>
                </div>
                
                <div class="section-title">💼 Current Portfolio</div>
                <div class="portfolio-grid">
                    {self._generate_portfolio_html(portfolio)}
                </div>
                
                <div class="section-title">📋 Recent Trades</div>
                {self._generate_trades_table(trade_history[-20:])}
                
                <div class="section-title">⚡ Advanced Metrics</div>
                <table class="trade-table">
                    <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                    <tr><td>Sharpe Ratio</td><td>{metrics.sharpe_ratio:.2f}</td><td>Risk-adjusted return</td></tr>
                    <tr><td>Maximum Drawdown</td><td>${metrics.max_drawdown:.2f}</td><td>Largest peak-to-trough decline</td></tr>
                    <tr><td>Average Hold Time</td><td>{metrics.average_hold_time:.1f} hours</td><td>Average time holding positions</td></tr>
                    <tr><td>Trades per Day</td><td>{metrics.trades_per_day:.1f}</td><td>Average daily trading frequency</td></tr>
                    <tr><td>Best Trade</td><td>${metrics.best_trade_profit:.2f}</td><td>Most profitable single trade</td></tr>
                    <tr><td>Worst Trade</td><td>${metrics.worst_trade_loss:.2f}</td><td>Largest single loss</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_chart_html(self, chart_paths: Dict[str, str]) -> str:
        """Generate HTML for charts"""
        chart_html = ""
        for chart_name, chart_path in chart_paths.items():
            if chart_path and Path(chart_path).exists():
                chart_html += f"""
                <div class="chart-container">
                    <img src="{chart_path}" alt="{chart_name.replace('_', ' ').title()}" />
                </div>
                """
        return chart_html
    
    def _generate_portfolio_html(self, portfolio: List[PortfolioItem]) -> str:
        """Generate HTML for portfolio items"""
        if not portfolio:
            return "<div class='portfolio-item'><h4>Portfolio is empty</h4></div>"
        
        html = ""
        for item in portfolio[:10]:  # Show top 10 items
            profit_class = "positive" if item.unrealized_profit > 0 else "negative"
            html += f"""
            <div class="portfolio-item">
                <h4>{item.item_name}</h4>
                <p>Purchase Price: ${item.purchase_price:.2f}</p>
                <p>Current Price: ${item.current_market_price:.2f}</p>
                <p class="{profit_class}">P&L: ${item.unrealized_profit:.2f} ({item.unrealized_profit_percentage:.1f}%)</p>
                <p>Days Held: {item.days_held}</p>
            </div>
            """
        return html
    
    def _generate_trades_table(self, trades: List[TradeRecord]) -> str:
        """Generate HTML table for recent trades"""
        if not trades:
            return "<p>No recent trades to display</p>"
        
        html = """
        <table class="trade-table">
            <tr>
                <th>Date</th>
                <th>Item</th>
                <th>Type</th>
                <th>Buy Price</th>
                <th>Target Price</th>
                <th>Profit</th>
                <th>Status</th>
            </tr>
        """
        
        for trade in reversed(trades):  # Most recent first
            profit_class = ""
            if trade.actual_profit:
                profit_class = "positive" if trade.actual_profit > 0 else "negative"
            
            html += f"""
            <tr>
                <td>{trade.created_at.strftime('%m/%d/%Y')}</td>
                <td>{trade.opportunity.item_name[:30]}...</td>
                <td>{trade.opportunity.trade_type.value}</td>
                <td>${trade.opportunity.market_price:.2f}</td>
                <td>${trade.opportunity.target_price:.2f}</td>
                <td class="{profit_class}">${trade.actual_profit:.2f if trade.actual_profit else 0:.2f}</td>
                <td>{trade.status.value}</td>
            </tr>
            """
        
        html += "</table>"
        return html

# Test function
async def test_trade_analytics():
    """Test trade analytics functionality"""
    print("🧪 Testing Trade Analytics...")
    
    # Create mock trading system
    mock_trading_system = None  # Would normally be a TradingSystem instance
    
    analytics = TradeAnalytics(mock_trading_system)
    
    # Test performance metrics calculation
    print("Test 1: Performance Metrics")
    mock_trades = []  # Would contain actual TradeRecord objects
    mock_portfolio = []  # Would contain actual PortfolioItem objects
    
    metrics = analytics.calculate_performance_metrics(mock_trades, mock_portfolio)
    print(f"✅ Performance metrics calculated: {metrics.total_trades} trades")
    
    print("✅ Trade Analytics test completed successfully")

if __name__ == "__main__":
    asyncio.run(test_trade_analytics())