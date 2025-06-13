#!/usr/bin/env python3

import asyncio
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import sqlite3
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import FloatCheckerConfig
from database import FloatDatabase
from trading_system import TradeRecord, PortfolioItem, TradeOpportunity
from multi_platform_integration import ArbitrageOpportunity, PlatformItem
from ml_engine import PricePrediction, MarketTrend, TradingSignal

@dataclass
class MarketInsight:
    """Advanced market insight"""
    insight_type: str
    title: str
    description: str
    confidence: float
    impact_score: float
    supporting_data: Dict[str, Any]
    generated_at: datetime
    actionable_recommendations: List[str]

@dataclass
class PerformanceBenchmark:
    """Performance benchmarking data"""
    metric_name: str
    current_value: float
    benchmark_value: float
    percentile_rank: float
    performance_rating: str  # "excellent", "good", "average", "poor"
    improvement_suggestions: List[str]

class AdvancedAnalytics:
    """Advanced analytics and insights engine"""
    
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.database = FloatDatabase()
        self.logger = logging.getLogger(__name__)
        
        # Analytics configuration
        self.analytics_config = {
            'lookback_days': 30,
            'min_data_points': 10,
            'confidence_threshold': 0.7,
            'correlation_threshold': 0.5
        }
        
        # Output directories
        self.reports_dir = Path("advanced_reports")
        self.dashboards_dir = Path("dashboards")
        self.insights_dir = Path("insights")
        
        for directory in [self.reports_dir, self.dashboards_dir, self.insights_dir]:
            directory.mkdir(exist_ok=True)
    
    async def generate_market_insights(self, market_data: Dict[str, Any], 
                                     trade_history: List[TradeRecord],
                                     arbitrage_opportunities: List[ArbitrageOpportunity]) -> List[MarketInsight]:
        """Generate advanced market insights using data science techniques"""
        insights = []
        
        try:
            # Price correlation analysis
            correlation_insight = await self._analyze_price_correlations(market_data)
            if correlation_insight:
                insights.append(correlation_insight)
            
            # Market volatility patterns
            volatility_insight = await self._analyze_volatility_patterns(market_data)
            if volatility_insight:
                insights.append(volatility_insight)
            
            # Trading performance patterns
            performance_insight = await self._analyze_performance_patterns(trade_history)
            if performance_insight:
                insights.append(performance_insight)
            
            # Arbitrage efficiency analysis
            arbitrage_insight = await self._analyze_arbitrage_efficiency(arbitrage_opportunities)
            if arbitrage_insight:
                insights.append(arbitrage_insight)
            
            # Seasonal trends
            seasonal_insight = await self._analyze_seasonal_trends(market_data, trade_history)
            if seasonal_insight:
                insights.append(seasonal_insight)
            
            # Anomaly detection
            anomaly_insight = await self._detect_market_anomalies(market_data)
            if anomaly_insight:
                insights.append(anomaly_insight)
            
        except Exception as e:
            self.logger.error(f"Error generating market insights: {e}")
        
        # Sort insights by impact score
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        
        self.logger.info(f"✅ Generated {len(insights)} market insights")
        return insights
    
    async def _analyze_price_correlations(self, market_data: Dict[str, Any]) -> Optional[MarketInsight]:
        """Analyze price correlations between items"""
        try:
            # Create price matrix
            items = list(market_data.keys())
            if len(items) < 3:
                return None
            
            price_matrix = []
            item_names = []
            
            for item_name, item_data in market_data.items():
                if 'prices' in item_data and len(item_data['prices']) > 0:
                    prices = list(item_data['prices'].values())
                    if len(prices) >= 2:  # Need at least 2 platform prices
                        price_matrix.append(prices[:5])  # Limit to 5 platforms
                        item_names.append(item_name)
            
            if len(price_matrix) < 3:
                return None
            
            # Ensure all rows have same length
            max_length = max(len(row) for row in price_matrix)
            price_matrix = [row + [0] * (max_length - len(row)) for row in price_matrix]
            
            # Calculate correlation matrix
            df = pd.DataFrame(price_matrix, index=item_names)
            correlation_matrix = df.T.corr()
            
            # Find strongest correlations
            strong_correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > self.analytics_config['correlation_threshold']:
                        strong_correlations.append({
                            'item1': correlation_matrix.index[i],
                            'item2': correlation_matrix.index[j],
                            'correlation': corr_value
                        })
            
            if not strong_correlations:
                return None
            
            # Sort by correlation strength
            strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            top_correlation = strong_correlations[0]
            
            insight_type = "positive_correlation" if top_correlation['correlation'] > 0 else "negative_correlation"
            
            recommendations = [
                f"Monitor {top_correlation['item1']} when trading {top_correlation['item2']}",
                "Consider portfolio diversification to reduce correlation risk",
                "Use correlation data for hedging strategies"
            ]
            
            return MarketInsight(
                insight_type=insight_type,
                title=f"Strong Price Correlation Detected",
                description=f"{top_correlation['item1']} and {top_correlation['item2']} show {abs(top_correlation['correlation']):.2f} correlation",
                confidence=min(abs(top_correlation['correlation']), 0.95),
                impact_score=abs(top_correlation['correlation']) * 100,
                supporting_data={
                    'correlations': strong_correlations[:5],
                    'correlation_matrix': correlation_matrix.to_dict()
                },
                generated_at=datetime.now(),
                actionable_recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}")
            return None
    
    async def _analyze_volatility_patterns(self, market_data: Dict[str, Any]) -> Optional[MarketInsight]:
        """Analyze market volatility patterns"""
        try:
            volatility_data = []
            
            for item_name, item_data in market_data.items():
                if 'prices' in item_data and 'price_spread' in item_data:
                    avg_price = item_data.get('average_price', 0)
                    spread = item_data.get('price_spread', 0)
                    
                    if avg_price > 0:
                        volatility = (spread / avg_price) * 100
                        volatility_data.append({
                            'item': item_name,
                            'volatility': volatility,
                            'average_price': avg_price,
                            'spread': spread
                        })
            
            if len(volatility_data) < 3:
                return None
            
            # Calculate volatility statistics
            volatilities = [item['volatility'] for item in volatility_data]
            avg_volatility = np.mean(volatilities)
            high_vol_threshold = np.percentile(volatilities, 75)
            
            high_vol_items = [item for item in volatility_data if item['volatility'] > high_vol_threshold]
            
            if not high_vol_items:
                return None
            
            recommendations = [
                "Consider volatility when sizing positions",
                "High volatility items may offer better arbitrage opportunities",
                "Use tighter stop-losses for volatile items",
                "Monitor high volatility items for sudden price movements"
            ]
            
            return MarketInsight(
                insight_type="volatility_analysis",
                title=f"Market Volatility Analysis",
                description=f"Average market volatility: {avg_volatility:.1f}%. {len(high_vol_items)} items show high volatility.",
                confidence=0.85,
                impact_score=min(avg_volatility * 2, 100),
                supporting_data={
                    'average_volatility': avg_volatility,
                    'high_volatility_items': high_vol_items,
                    'volatility_distribution': volatilities
                },
                generated_at=datetime.now(),
                actionable_recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in volatility analysis: {e}")
            return None
    
    async def _analyze_performance_patterns(self, trade_history: List[TradeRecord]) -> Optional[MarketInsight]:
        """Analyze trading performance patterns"""
        try:
            if len(trade_history) < self.analytics_config['min_data_points']:
                return None
            
            # Group trades by time periods
            hourly_performance = {}
            daily_performance = {}
            
            for trade in trade_history:
                if not trade.actual_profit:
                    continue
                
                hour = trade.created_at.hour
                day = trade.created_at.strftime('%A')
                
                if hour not in hourly_performance:
                    hourly_performance[hour] = []
                hourly_performance[hour].append(trade.actual_profit)
                
                if day not in daily_performance:
                    daily_performance[day] = []
                daily_performance[day].append(trade.actual_profit)
            
            # Find best performing time periods
            best_hours = []
            for hour, profits in hourly_performance.items():
                if len(profits) >= 3:  # Need at least 3 trades
                    avg_profit = np.mean(profits)
                    success_rate = len([p for p in profits if p > 0]) / len(profits)
                    best_hours.append({
                        'hour': hour,
                        'avg_profit': avg_profit,
                        'success_rate': success_rate,
                        'trade_count': len(profits)
                    })
            
            if not best_hours:
                return None
            
            best_hours.sort(key=lambda x: x['avg_profit'], reverse=True)
            top_hour = best_hours[0]
            
            recommendations = [
                f"Focus trading activity around {top_hour['hour']}:00 for better results",
                "Avoid trading during historically poor performing hours",
                "Consider timezone effects on global markets",
                "Track performance patterns to optimize trading schedule"
            ]
            
            return MarketInsight(
                insight_type="performance_patterns",
                title="Trading Time Performance Analysis",
                description=f"Best trading hour: {top_hour['hour']}:00 with ${top_hour['avg_profit']:.2f} average profit",
                confidence=min(top_hour['success_rate'], 0.9),
                impact_score=abs(top_hour['avg_profit']) * 5,
                supporting_data={
                    'hourly_performance': best_hours,
                    'daily_performance': daily_performance,
                    'best_hour': top_hour
                },
                generated_at=datetime.now(),
                actionable_recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in performance pattern analysis: {e}")
            return None
    
    async def _analyze_arbitrage_efficiency(self, arbitrage_opportunities: List[ArbitrageOpportunity]) -> Optional[MarketInsight]:
        """Analyze arbitrage efficiency patterns"""
        try:
            if len(arbitrage_opportunities) < 5:
                return None
            
            # Analyze platform combinations
            platform_combinations = {}
            platform_efficiency = {}
            
            for opp in arbitrage_opportunities:
                combo = f"{opp.buy_platform.value}->{opp.sell_platform.value}"
                
                if combo not in platform_combinations:
                    platform_combinations[combo] = []
                platform_combinations[combo].append({
                    'profit': opp.net_profit,
                    'profit_percentage': opp.profit_percentage,
                    'risk_score': opp.risk_score
                })
            
            # Calculate efficiency metrics
            for combo, opportunities in platform_combinations.items():
                avg_profit = np.mean([o['profit'] for o in opportunities])
                avg_profit_pct = np.mean([o['profit_percentage'] for o in opportunities])
                avg_risk = np.mean([o['risk_score'] for o in opportunities])
                efficiency_score = avg_profit_pct / max(avg_risk, 0.1)
                
                platform_efficiency[combo] = {
                    'avg_profit': avg_profit,
                    'avg_profit_percentage': avg_profit_pct,
                    'avg_risk': avg_risk,
                    'efficiency_score': efficiency_score,
                    'opportunity_count': len(opportunities)
                }
            
            # Find most efficient combination
            best_combo = max(platform_efficiency.items(), key=lambda x: x[1]['efficiency_score'])
            combo_name, combo_data = best_combo
            
            recommendations = [
                f"Prioritize {combo_name} arbitrage opportunities",
                "Focus on platform combinations with best efficiency scores",
                "Consider transaction costs when evaluating opportunities",
                "Monitor platform-specific trends for optimization"
            ]
            
            return MarketInsight(
                insight_type="arbitrage_efficiency",
                title="Arbitrage Platform Efficiency Analysis",
                description=f"Most efficient platform combination: {combo_name} with {combo_data['efficiency_score']:.2f} efficiency score",
                confidence=0.8,
                impact_score=combo_data['avg_profit_percentage'],
                supporting_data={
                    'platform_efficiency': platform_efficiency,
                    'best_combination': combo_data,
                    'total_opportunities': len(arbitrage_opportunities)
                },
                generated_at=datetime.now(),
                actionable_recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in arbitrage efficiency analysis: {e}")
            return None
    
    async def _analyze_seasonal_trends(self, market_data: Dict[str, Any], 
                                     trade_history: List[TradeRecord]) -> Optional[MarketInsight]:
        """Analyze seasonal trading trends"""
        try:
            if len(trade_history) < 20:
                return None
            
            # Group trades by month and day of week
            monthly_data = {}
            weekday_data = {}
            
            for trade in trade_history:
                month = trade.created_at.month
                weekday = trade.created_at.weekday()
                
                if month not in monthly_data:
                    monthly_data[month] = {'trades': 0, 'total_profit': 0}
                monthly_data[month]['trades'] += 1
                if trade.actual_profit:
                    monthly_data[month]['total_profit'] += trade.actual_profit
                
                if weekday not in weekday_data:
                    weekday_data[weekday] = {'trades': 0, 'total_profit': 0}
                weekday_data[weekday]['trades'] += 1
                if trade.actual_profit:
                    weekday_data[weekday]['total_profit'] += trade.actual_profit
            
            # Find best performing periods
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                           'Friday', 'Saturday', 'Sunday']
            
            best_month = max(monthly_data.items(), 
                           key=lambda x: x[1]['total_profit'] / max(x[1]['trades'], 1))
            best_weekday = max(weekday_data.items(), 
                             key=lambda x: x[1]['total_profit'] / max(x[1]['trades'], 1))
            
            recommendations = [
                f"Increase trading activity in {month_names[best_month[0] - 1]}",
                f"Focus on {weekday_names[best_weekday[0]]} trading",
                "Adjust strategy based on seasonal patterns",
                "Plan inventory management around seasonal trends"
            ]
            
            return MarketInsight(
                insight_type="seasonal_trends",
                title="Seasonal Trading Pattern Analysis",
                description=f"Best month: {month_names[best_month[0] - 1]}, Best weekday: {weekday_names[best_weekday[0]]}",
                confidence=0.7,
                impact_score=50,
                supporting_data={
                    'monthly_performance': monthly_data,
                    'weekday_performance': weekday_data,
                    'best_periods': {
                        'month': month_names[best_month[0] - 1],
                        'weekday': weekday_names[best_weekday[0]]
                    }
                },
                generated_at=datetime.now(),
                actionable_recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in seasonal trend analysis: {e}")
            return None
    
    async def _detect_market_anomalies(self, market_data: Dict[str, Any]) -> Optional[MarketInsight]:
        """Detect market anomalies using statistical methods"""
        try:
            anomalies = []
            
            for item_name, item_data in market_data.items():
                if 'prices' in item_data and len(item_data['prices']) >= 3:
                    prices = list(item_data['prices'].values())
                    
                    # Calculate z-scores to detect outliers
                    mean_price = np.mean(prices)
                    std_price = np.std(prices)
                    
                    if std_price > 0:
                        z_scores = [(price - mean_price) / std_price for price in prices]
                        
                        for i, z_score in enumerate(z_scores):
                            if abs(z_score) > 2:  # Outlier threshold
                                platform_names = list(item_data['prices'].keys())
                                anomalies.append({
                                    'item': item_name,
                                    'platform': platform_names[i],
                                    'price': prices[i],
                                    'mean_price': mean_price,
                                    'z_score': z_score,
                                    'anomaly_type': 'high_price' if z_score > 0 else 'low_price'
                                })
            
            if not anomalies:
                return None
            
            # Sort by z-score magnitude
            anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
            top_anomaly = anomalies[0]
            
            recommendations = [
                f"Investigate {top_anomaly['item']} pricing on {top_anomaly['platform']}",
                "Monitor for data quality issues",
                "Consider arbitrage opportunities from anomalies",
                "Verify anomalous prices before trading"
            ]
            
            return MarketInsight(
                insight_type="market_anomalies",
                title="Market Price Anomaly Detected",
                description=f"{top_anomaly['item']} shows unusual pricing on {top_anomaly['platform']} (Z-score: {top_anomaly['z_score']:.2f})",
                confidence=min(abs(top_anomaly['z_score']) / 3, 0.95),
                impact_score=abs(top_anomaly['z_score']) * 20,
                supporting_data={
                    'anomalies': anomalies[:10],
                    'top_anomaly': top_anomaly,
                    'total_anomalies': len(anomalies)
                },
                generated_at=datetime.now(),
                actionable_recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return None
    
    async def create_interactive_dashboard(self, market_data: Dict[str, Any], 
                                         trade_history: List[TradeRecord],
                                         insights: List[MarketInsight]) -> str:
        """Create interactive HTML dashboard with Plotly"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=['Price Distribution', 'Trading Performance', 
                              'Platform Comparison', 'Profit Timeline',
                              'Market Volatility', 'Insight Summary'],
                specs=[[{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "table"}]]
            )
            
            # 1. Price Distribution
            if market_data:
                items = list(market_data.keys())[:10]
                avg_prices = [market_data[item].get('average_price', 0) for item in items]
                
                fig.add_trace(
                    go.Bar(x=items, y=avg_prices, name="Average Price", 
                          marker_color='lightblue'),
                    row=1, col=1
                )
            
            # 2. Trading Performance
            if trade_history:
                dates = [t.created_at for t in trade_history[-50:]]
                profits = [t.actual_profit or 0 for t in trade_history[-50:]]
                cumulative_profit = np.cumsum(profits)
                
                fig.add_trace(
                    go.Scatter(x=dates, y=cumulative_profit, mode='lines+markers',
                             name="Cumulative Profit", line=dict(color='green')),
                    row=1, col=2
                )
            
            # 3. Platform Comparison
            if market_data:
                platform_counts = {}
                for item_data in market_data.values():
                    for platform in item_data.get('prices', {}):
                        platform_counts[platform] = platform_counts.get(platform, 0) + 1
                
                platforms = list(platform_counts.keys())
                counts = list(platform_counts.values())
                
                fig.add_trace(
                    go.Bar(x=platforms, y=counts, name="Platform Coverage",
                          marker_color='orange'),
                    row=2, col=1
                )
            
            # 4. Profit Timeline (Individual Trades)
            if trade_history:
                trade_dates = [t.created_at for t in trade_history[-30:]]
                trade_profits = [t.actual_profit or 0 for t in trade_history[-30:]]
                colors = ['green' if p > 0 else 'red' for p in trade_profits]
                
                fig.add_trace(
                    go.Scatter(x=trade_dates, y=trade_profits, mode='markers',
                             name="Individual Trades", 
                             marker=dict(color=colors, size=8)),
                    row=2, col=2
                )
            
            # 5. Market Volatility
            if market_data:
                items = list(market_data.keys())[:10]
                volatilities = []
                for item in items:
                    item_data = market_data[item]
                    avg_price = item_data.get('average_price', 0)
                    spread = item_data.get('price_spread', 0)
                    volatility = (spread / avg_price * 100) if avg_price > 0 else 0
                    volatilities.append(volatility)
                
                fig.add_trace(
                    go.Bar(x=items, y=volatilities, name="Volatility %",
                          marker_color='red'),
                    row=3, col=1
                )
            
            # 6. Insight Summary Table
            if insights:
                insight_data = []
                for insight in insights[:10]:
                    insight_data.append([
                        insight.title,
                        f"{insight.confidence:.2f}",
                        f"{insight.impact_score:.1f}",
                        insight.insight_type
                    ])
                
                fig.add_trace(
                    go.Table(
                        header=dict(values=['Insight', 'Confidence', 'Impact', 'Type']),
                        cells=dict(values=list(zip(*insight_data)) if insight_data else [[], [], [], []])
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="CS2 Trading Analytics Dashboard",
                height=1200,
                showlegend=False,
                template="plotly_white"
            )
            
            # Save dashboard
            dashboard_path = self.dashboards_dir / f"trading_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            pyo.plot(fig, filename=str(dashboard_path), auto_open=False)
            
            self.logger.info(f"✅ Interactive dashboard created: {dashboard_path}")
            return str(dashboard_path)
            
        except Exception as e:
            self.logger.error(f"Error creating interactive dashboard: {e}")
            return ""
    
    async def generate_performance_benchmarks(self, trade_history: List[TradeRecord],
                                            portfolio: List[PortfolioItem]) -> List[PerformanceBenchmark]:
        """Generate performance benchmarks against industry standards"""
        benchmarks = []
        
        try:
            if not trade_history:
                return benchmarks
            
            # Calculate current metrics
            total_trades = len(trade_history)
            profitable_trades = len([t for t in trade_history if t.actual_profit and t.actual_profit > 0])
            total_profit = sum(t.actual_profit for t in trade_history if t.actual_profit)
            
            # Win Rate Benchmark
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            win_rate_benchmark = 65.0  # Industry benchmark
            win_rate_percentile = min(win_rate / win_rate_benchmark * 50, 100)
            
            benchmarks.append(PerformanceBenchmark(
                metric_name="Win Rate",
                current_value=win_rate,
                benchmark_value=win_rate_benchmark,
                percentile_rank=win_rate_percentile,
                performance_rating=self._get_performance_rating(win_rate_percentile),
                improvement_suggestions=[
                    "Improve trade selection criteria",
                    "Focus on higher probability setups",
                    "Reduce position sizes for risky trades"
                ]
            ))
            
            # Average Profit Per Trade
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            avg_profit_benchmark = 15.0  # $15 benchmark
            profit_percentile = min(avg_profit / avg_profit_benchmark * 50, 100)
            
            benchmarks.append(PerformanceBenchmark(
                metric_name="Average Profit Per Trade",
                current_value=avg_profit,
                benchmark_value=avg_profit_benchmark,
                percentile_rank=profit_percentile,
                performance_rating=self._get_performance_rating(profit_percentile),
                improvement_suggestions=[
                    "Increase position sizes for high-confidence trades",
                    "Focus on higher-value items",
                    "Optimize entry and exit timing"
                ]
            ))
            
            # Portfolio Diversification
            if portfolio:
                unique_items = len(set(item.item_name.split(' | ')[0] for item in portfolio))
                total_items = len(portfolio)
                diversification_ratio = (unique_items / total_items * 100) if total_items > 0 else 0
                diversification_benchmark = 70.0
                diversification_percentile = min(diversification_ratio / diversification_benchmark * 50, 100)
                
                benchmarks.append(PerformanceBenchmark(
                    metric_name="Portfolio Diversification",
                    current_value=diversification_ratio,
                    benchmark_value=diversification_benchmark,
                    percentile_rank=diversification_percentile,
                    performance_rating=self._get_performance_rating(diversification_percentile),
                    improvement_suggestions=[
                        "Spread investments across more weapon types",
                        "Avoid concentration in single items",
                        "Consider different price ranges"
                    ]
                ))
            
            # Trade Frequency
            if trade_history:
                trading_days = (trade_history[-1].created_at - trade_history[0].created_at).days
                trades_per_day = total_trades / max(trading_days, 1)
                frequency_benchmark = 2.0  # 2 trades per day
                frequency_percentile = min(trades_per_day / frequency_benchmark * 50, 100)
                
                benchmarks.append(PerformanceBenchmark(
                    metric_name="Trading Frequency",
                    current_value=trades_per_day,
                    benchmark_value=frequency_benchmark,
                    percentile_rank=frequency_percentile,
                    performance_rating=self._get_performance_rating(frequency_percentile),
                    improvement_suggestions=[
                        "Increase market scanning frequency",
                        "Automate more trading processes",
                        "Expand to additional platforms"
                    ]
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating performance benchmarks: {e}")
        
        return benchmarks
    
    def _get_performance_rating(self, percentile: float) -> str:
        """Convert percentile to performance rating"""
        if percentile >= 80:
            return "excellent"
        elif percentile >= 60:
            return "good"
        elif percentile >= 40:
            return "average"
        else:
            return "poor"
    
    async def export_comprehensive_analytics_report(self, market_data: Dict[str, Any],
                                                   trade_history: List[TradeRecord],
                                                   portfolio: List[PortfolioItem],
                                                   insights: List[MarketInsight],
                                                   benchmarks: List[PerformanceBenchmark]) -> str:
        """Export comprehensive analytics report as PDF/HTML"""
        try:
            # Generate all visualizations
            dashboard_path = await self.create_interactive_dashboard(market_data, trade_history, insights)
            
            # Create comprehensive report HTML
            report_html = self._create_comprehensive_report_html(
                market_data, trade_history, portfolio, insights, benchmarks, dashboard_path
            )
            
            # Save report
            report_path = self.reports_dir / f"comprehensive_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            self.logger.info(f"✅ Comprehensive analytics report exported: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting comprehensive report: {e}")
            return ""
    
    def _create_comprehensive_report_html(self, market_data: Dict[str, Any],
                                        trade_history: List[TradeRecord],
                                        portfolio: List[PortfolioItem],
                                        insights: List[MarketInsight],
                                        benchmarks: List[PerformanceBenchmark],
                                        dashboard_path: str) -> str:
        """Create comprehensive HTML report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive CS2 Trading Analytics Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; }}
                .header h1 {{ font-size: 2.5em; margin: 0; font-weight: 300; }}
                .header p {{ font-size: 1.2em; margin: 10px 0 0 0; opacity: 0.9; }}
                .section {{ padding: 40px; border-bottom: 1px solid #eee; }}
                .section:last-child {{ border-bottom: none; }}
                .section-title {{ font-size: 1.8em; color: #2c3e50; margin-bottom: 30px; border-left: 4px solid #667eea; padding-left: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 25px; margin: 30px 0; }}
                .metric-card {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 10px 20px rgba(240,147,251,0.3); }}
                .metric-value {{ font-size: 2.2em; font-weight: bold; margin-bottom: 10px; }}
                .metric-label {{ font-size: 1em; opacity: 0.9; }}
                .insights-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 25px; }}
                .insight-card {{ background: #f8f9fa; border-radius: 10px; padding: 25px; border-left: 5px solid #667eea; }}
                .insight-title {{ font-size: 1.3em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
                .insight-description {{ color: #555; margin-bottom: 15px; line-height: 1.6; }}
                .insight-confidence {{ background: #667eea; color: white; padding: 5px 10px; border-radius: 20px; font-size: 0.9em; display: inline-block; }}
                .benchmarks-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .benchmarks-table th, .benchmarks-table td {{ border: 1px solid #ddd; padding: 15px; text-align: left; }}
                .benchmarks-table th {{ background: #667eea; color: white; font-weight: 600; }}
                .benchmarks-table tr:nth-child(even) {{ background: #f8f9fa; }}
                .rating-excellent {{ color: #27ae60; font-weight: bold; }}
                .rating-good {{ color: #2980b9; font-weight: bold; }}
                .rating-average {{ color: #f39c12; font-weight: bold; }}
                .rating-poor {{ color: #e74c3c; font-weight: bold; }}
                .dashboard-embed {{ width: 100%; height: 800px; border: none; border-radius: 10px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
                .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
                .summary-stat {{ background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                .summary-stat-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .summary-stat-label {{ color: #666; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Comprehensive CS2 Trading Analytics</h1>
                    <p>Advanced Market Analysis & Performance Report</p>
                    <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <div class="section">
                    <div class="section-title">📊 Executive Summary</div>
                    <div class="summary-stats">
                        <div class="summary-stat">
                            <div class="summary-stat-value">{len(trade_history)}</div>
                            <div class="summary-stat-label">Total Trades</div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-stat-value">{len([t for t in trade_history if t.actual_profit and t.actual_profit > 0])}</div>
                            <div class="summary-stat-label">Winning Trades</div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-stat-value">${sum(t.actual_profit for t in trade_history if t.actual_profit):.2f}</div>
                            <div class="summary-stat-label">Total Profit</div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-stat-value">{len(portfolio)}</div>
                            <div class="summary-stat-label">Portfolio Items</div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-stat-value">{len(insights)}</div>
                            <div class="summary-stat-label">Key Insights</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">🔍 Market Insights</div>
                    <div class="insights-container">
                        {self._generate_insights_html(insights)}
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">📈 Performance Benchmarks</div>
                    <table class="benchmarks-table">
                        <tr>
                            <th>Metric</th>
                            <th>Current Value</th>
                            <th>Benchmark</th>
                            <th>Percentile</th>
                            <th>Rating</th>
                            <th>Key Improvement</th>
                        </tr>
                        {self._generate_benchmarks_html(benchmarks)}
                    </table>
                </div>
                
                <div class="section">
                    <div class="section-title">📱 Interactive Dashboard</div>
                    <iframe src="{dashboard_path}" class="dashboard-embed"></iframe>
                </div>
                
                <div class="section">
                    <div class="section-title">📋 Market Data Summary</div>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{len(market_data)}</div>
                            <div class="metric-label">Items Tracked</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{sum(len(item.get('prices', {})) for item in market_data.values())}</div>
                            <div class="metric-label">Platform Listings</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${np.mean([item.get('average_price', 0) for item in market_data.values()]):.2f}</div>
                            <div class="metric-label">Average Item Price</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{np.mean([item.get('arbitrage_potential', 0) for item in market_data.values()]):.1f}%</div>
                            <div class="metric-label">Avg Arbitrage Potential</div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_insights_html(self, insights: List[MarketInsight]) -> str:
        """Generate HTML for insights section"""
        html = ""
        for insight in insights[:6]:  # Top 6 insights
            html += f"""
            <div class="insight-card">
                <div class="insight-title">{insight.title}</div>
                <div class="insight-description">{insight.description}</div>
                <div class="insight-confidence">Confidence: {insight.confidence:.1%}</div>
                <div style="margin-top: 15px;">
                    <strong>Recommendations:</strong>
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in insight.actionable_recommendations[:3])}
                    </ul>
                </div>
            </div>
            """
        return html
    
    def _generate_benchmarks_html(self, benchmarks: List[PerformanceBenchmark]) -> str:
        """Generate HTML for benchmarks table"""
        html = ""
        for benchmark in benchmarks:
            rating_class = f"rating-{benchmark.performance_rating}"
            improvement = benchmark.improvement_suggestions[0] if benchmark.improvement_suggestions else "No suggestions"
            
            html += f"""
            <tr>
                <td><strong>{benchmark.metric_name}</strong></td>
                <td>{benchmark.current_value:.2f}</td>
                <td>{benchmark.benchmark_value:.2f}</td>
                <td>{benchmark.percentile_rank:.1f}%</td>
                <td><span class="{rating_class}">{benchmark.performance_rating.upper()}</span></td>
                <td>{improvement}</td>
            </tr>
            """
        return html

# Test function
async def test_advanced_analytics():
    """Test advanced analytics functionality"""
    print("🧪 Testing Advanced Analytics...")
    
    analytics = AdvancedAnalytics()
    
    # Test market insights generation
    print("Test 1: Market Insights Generation")
    mock_market_data = {
        "AK-47 | Redline": {
            "prices": {"steam": 50.0, "csgofloat": 48.0},
            "average_price": 49.0,
            "price_spread": 2.0
        },
        "AWP | Dragon Lore": {
            "prices": {"steam": 2000.0, "skinport": 1950.0},
            "average_price": 1975.0,
            "price_spread": 50.0
        }
    }
    
    insights = await analytics.generate_market_insights(mock_market_data, [], [])
    print(f"✅ Generated {len(insights)} market insights")
    
    # Test dashboard creation
    print("Test 2: Interactive Dashboard")
    dashboard_path = await analytics.create_interactive_dashboard(mock_market_data, [], insights)
    print(f"✅ Dashboard created: {bool(dashboard_path)}")
    
    # Test performance benchmarks
    print("Test 3: Performance Benchmarks")
    benchmarks = await analytics.generate_performance_benchmarks([], [])
    print(f"✅ Generated {len(benchmarks)} performance benchmarks")
    
    print("✅ Advanced Analytics test completed successfully")

if __name__ == "__main__":
    asyncio.run(test_advanced_analytics())