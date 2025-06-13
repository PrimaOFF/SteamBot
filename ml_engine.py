#!/usr/bin/env python3

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import FloatCheckerConfig
from database import FloatDatabase
from trading_system import TradeRecord, PortfolioItem, TradeOpportunity
from float_analyzer import FloatAnalysis

@dataclass
class PricePrediction:
    """Price prediction result"""
    item_name: str
    current_price: float
    predicted_price: float
    confidence: float
    timeframe_hours: int
    model_name: str
    features_used: List[str]
    prediction_date: datetime
    price_change_percentage: float = 0.0
    
    def __post_init__(self):
        if self.current_price > 0:
            self.price_change_percentage = ((self.predicted_price - self.current_price) / self.current_price) * 100

@dataclass
class MarketTrend:
    """Market trend analysis result"""
    trend_direction: str  # "bullish", "bearish", "sideways"
    strength: float  # 0-1 scale
    confidence: float
    duration_hours: int
    key_factors: List[str]
    analysis_date: datetime

@dataclass
class TradingSignal:
    """AI-generated trading signal"""
    item_name: str
    signal_type: str  # "buy", "sell", "hold"
    strength: float  # 0-1 scale
    confidence: float
    price_target: float
    stop_loss: float
    reasoning: List[str]
    generated_at: datetime
    valid_until: datetime

class FeatureEngineering:
    """Advanced feature engineering for ML models"""
    
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
    
    def extract_price_features(self, price_history: List[Tuple[datetime, float]]) -> Dict[str, float]:
        """Extract price-based features"""
        if len(price_history) < 2:
            return {}
        
        prices = [p[1] for p in price_history]
        dates = [p[0] for p in price_history]
        
        features = {}
        
        # Basic price statistics
        features['current_price'] = prices[-1]
        features['price_mean_7d'] = np.mean(prices[-7:]) if len(prices) >= 7 else np.mean(prices)
        features['price_mean_30d'] = np.mean(prices[-30:]) if len(prices) >= 30 else np.mean(prices)
        features['price_std_7d'] = np.std(prices[-7:]) if len(prices) >= 7 else np.std(prices)
        features['price_std_30d'] = np.std(prices[-30:]) if len(prices) >= 30 else np.std(prices)
        
        # Price momentum
        if len(prices) >= 7:
            features['momentum_7d'] = (prices[-1] - prices[-7]) / prices[-7] * 100
        if len(prices) >= 30:
            features['momentum_30d'] = (prices[-1] - prices[-30]) / prices[-30] * 100
        
        # Volatility
        if len(prices) >= 2:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            features['volatility'] = np.std(returns) * 100
        
        # Price channels
        if len(prices) >= 7:
            recent_high = max(prices[-7:])
            recent_low = min(prices[-7:])
            features['price_position'] = (prices[-1] - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
        
        # Trend strength
        if len(prices) >= 5:
            x = np.array(range(len(prices[-5:])))
            y = np.array(prices[-5:])
            slope = np.polyfit(x, y, 1)[0]
            features['trend_strength'] = slope / np.mean(prices[-5:])
        
        return features
    
    def extract_item_features(self, item_name: str) -> Dict[str, float]:
        """Extract item-specific features"""
        features = {}
        
        # Weapon type encoding
        weapon_types = ['AK-47', 'AWP', 'M4A4', 'M4A1-S', 'Glock-18', 'USP-S', 'Karambit', 'Butterfly']
        for weapon in weapon_types:
            features[f'is_{weapon.lower().replace("-", "_")}'] = 1.0 if weapon in item_name else 0.0
        
        # Skin rarity indicators
        rare_skins = ['Dragon Lore', 'Howl', 'Fire Serpent', 'Fade', 'Doppler', 'Case Hardened']
        for skin in rare_skins:
            features[f'is_{skin.lower().replace(" ", "_")}'] = 1.0 if skin in item_name else 0.0
        
        # Wear condition
        wear_conditions = ['Factory New', 'Minimal Wear', 'Field-Tested', 'Well-Worn', 'Battle-Scarred']
        for wear in wear_conditions:
            features[f'wear_{wear.lower().replace("-", "_").replace(" ", "_")}'] = 1.0 if wear in item_name else 0.0
        
        # Special attributes
        features['is_stattrak'] = 1.0 if 'StatTrak' in item_name else 0.0
        features['is_souvenir'] = 1.0 if 'Souvenir' in item_name else 0.0
        features['is_knife'] = 1.0 if any(knife in item_name for knife in ['Karambit', 'Butterfly', 'Bayonet', 'Flip']) else 0.0
        features['is_gloves'] = 1.0 if 'Gloves' in item_name else 0.0
        
        return features
    
    def extract_market_features(self, trade_history: List[TradeRecord]) -> Dict[str, float]:
        """Extract market-wide features"""
        features = {}
        
        if not trade_history:
            return features
        
        recent_trades = [t for t in trade_history if t.created_at > datetime.now() - timedelta(days=7)]
        
        # Market sentiment
        successful_trades = [t for t in recent_trades if t.actual_profit and t.actual_profit > 0]
        features['market_sentiment'] = len(successful_trades) / max(len(recent_trades), 1)
        
        # Trading volume
        features['trading_volume_7d'] = len(recent_trades)
        
        # Average profit
        profits = [t.actual_profit for t in recent_trades if t.actual_profit]
        features['avg_profit_7d'] = np.mean(profits) if profits else 0.0
        
        # Market volatility
        if profits:
            features['market_volatility'] = np.std(profits)
        
        return features
    
    def extract_float_features(self, float_value: Optional[float], item_name: str) -> Dict[str, float]:
        """Extract float-specific features"""
        features = {}
        
        if float_value is None:
            return features
        
        features['float_value'] = float_value
        features['float_log'] = np.log10(max(float_value, 0.00001))
        
        # Float rarity
        if 'Factory New' in item_name:
            features['float_rarity'] = 1.0 - (float_value / 0.07)  # Lower float = higher rarity for FN
        elif 'Battle-Scarred' in item_name:
            features['float_rarity'] = float_value / 1.0  # Higher float = higher rarity for BS
        else:
            features['float_rarity'] = 0.5  # Neutral for other wears
        
        # Float extremeness
        features['is_extreme_float'] = 1.0 if float_value <= 0.001 or float_value >= 0.999 else 0.0
        features['float_percentile'] = self._calculate_float_percentile(float_value, item_name)
        
        return features
    
    def _calculate_float_percentile(self, float_value: float, item_name: str) -> float:
        """Calculate float percentile within its wear range"""
        base_skin = item_name.split(' (')[0]
        skin_data = self.config.SKIN_SPECIFIC_RANGES.get(base_skin, {})
        
        if 'Factory New' in item_name:
            fn_range = skin_data.get('Factory New', (0.00, 0.07))
            return (float_value - fn_range[0]) / (fn_range[1] - fn_range[0])
        elif 'Battle-Scarred' in item_name:
            bs_range = skin_data.get('Battle-Scarred', (0.45, 1.00))
            return (float_value - bs_range[0]) / (bs_range[1] - bs_range[0])
        
        return 0.5  # Default for other wears
    
    def create_feature_vector(self, item_name: str, price_history: List[Tuple[datetime, float]], 
                            trade_history: List[TradeRecord], float_value: Optional[float] = None) -> np.ndarray:
        """Create complete feature vector for ML models"""
        features = {}
        
        # Combine all feature types
        features.update(self.extract_price_features(price_history))
        features.update(self.extract_item_features(item_name))
        features.update(self.extract_market_features(trade_history))
        features.update(self.extract_float_features(float_value, item_name))
        
        # Add time-based features
        now = datetime.now()
        features['hour_of_day'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = 1.0 if now.weekday() >= 5 else 0.0
        
        # Convert to array (ensure consistent ordering)
        feature_names = sorted(features.keys())
        feature_vector = np.array([features.get(name, 0.0) for name in feature_names])
        
        return feature_vector, feature_names

class MLPredictor:
    """Machine learning price prediction engine"""
    
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.database = FloatDatabase()
        self.feature_engine = FeatureEngineering()
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.model_metadata = {}
        self.models_dir = Path("ml_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Available models
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'description': 'Random Forest for robust predictions'
            },
            'gradient_boost': {
                'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'description': 'Gradient Boosting for complex patterns'
            },
            'linear_ridge': {
                'model': Ridge(alpha=1.0),
                'description': 'Ridge Regression for linear relationships'
            }
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        for name, config in self.model_configs.items():
            self.models[name] = config['model']
            self.model_metadata[name] = {
                'trained': False,
                'last_training': None,
                'accuracy_score': 0.0,
                'feature_names': [],
                'training_samples': 0
            }
    
    async def train_models(self, training_data: List[Dict]) -> Dict[str, float]:
        """Train all ML models with historical data"""
        self.logger.info("🤖 Starting ML model training...")
        
        if len(training_data) < 50:
            self.logger.warning("Insufficient training data (need at least 50 samples)")
            return {}
        
        # Prepare training dataset
        X, y, feature_names = self._prepare_training_data(training_data)
        
        if X.shape[0] == 0:
            self.logger.error("No valid training samples prepared")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        training_results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                cv_mean = np.mean(cv_scores)
                
                # Update metadata
                self.model_metadata[model_name].update({
                    'trained': True,
                    'last_training': datetime.now(),
                    'accuracy_score': r2,
                    'feature_names': feature_names,
                    'training_samples': len(X_train),
                    'mae': mae,
                    'rmse': rmse,
                    'cv_score': cv_mean
                })
                
                training_results[model_name] = r2
                
                # Save model
                model_path = self.models_dir / f"{model_name}.joblib"
                joblib.dump({
                    'model': model,
                    'scaler': scaler,
                    'metadata': self.model_metadata[model_name]
                }, model_path)
                
                self.logger.info(f"✅ {model_name}: R² = {r2:.3f}, MAE = ${mae:.2f}, RMSE = ${rmse:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = 0.0
        
        self.logger.info(f"🎯 Model training completed. Best model: {max(training_results, key=training_results.get)}")
        return training_results
    
    def _prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training dataset from historical data"""
        X_list = []
        y_list = []
        feature_names = None
        
        for data_point in training_data:
            try:
                # Extract features
                feature_vector, names = self.feature_engine.create_feature_vector(
                    item_name=data_point['item_name'],
                    price_history=data_point['price_history'],
                    trade_history=data_point['trade_history'],
                    float_value=data_point.get('float_value')
                )
                
                target_price = data_point['target_price']
                
                if feature_names is None:
                    feature_names = names
                
                X_list.append(feature_vector)
                y_list.append(target_price)
                
            except Exception as e:
                self.logger.error(f"Error processing training sample: {e}")
                continue
        
        if not X_list:
            return np.array([]), np.array([]), []
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Remove invalid samples
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"Prepared {len(X)} valid training samples with {X.shape[1]} features")
        return X, y, feature_names
    
    async def predict_price(self, item_name: str, price_history: List[Tuple[datetime, float]], 
                          trade_history: List[TradeRecord], float_value: Optional[float] = None,
                          timeframe_hours: int = 24) -> List[PricePrediction]:
        """Predict future price using trained models"""
        predictions = []
        
        try:
            # Create feature vector
            feature_vector, feature_names = self.feature_engine.create_feature_vector(
                item_name, price_history, trade_history, float_value
            )
            
            current_price = price_history[-1][1] if price_history else 0.0
            
            # Get predictions from each trained model
            for model_name, model in self.models.items():
                metadata = self.model_metadata[model_name]
                
                if not metadata['trained']:
                    continue
                
                try:
                    # Load model if needed
                    model_path = self.models_dir / f"{model_name}.joblib"
                    if model_path.exists():
                        saved_data = joblib.load(model_path)
                        model = saved_data['model']
                        scaler = saved_data['scaler']
                        
                        # Scale features
                        feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))
                        
                        # Predict
                        predicted_price = model.predict(feature_vector_scaled)[0]
                        
                        # Calculate confidence based on model accuracy
                        confidence = min(metadata['accuracy_score'], 0.95)
                        
                        prediction = PricePrediction(
                            item_name=item_name,
                            current_price=current_price,
                            predicted_price=max(predicted_price, 0.01),  # Ensure positive price
                            confidence=confidence,
                            timeframe_hours=timeframe_hours,
                            model_name=model_name,
                            features_used=feature_names,
                            prediction_date=datetime.now()
                        )
                        
                        predictions.append(prediction)
                        
                except Exception as e:
                    self.logger.error(f"Error getting prediction from {model_name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in price prediction: {e}")
        
        return predictions
    
    def get_ensemble_prediction(self, predictions: List[PricePrediction]) -> Optional[PricePrediction]:
        """Create ensemble prediction from multiple models"""
        if not predictions:
            return None
        
        # Weight predictions by confidence
        total_weight = sum(p.confidence for p in predictions)
        if total_weight == 0:
            return None
        
        weighted_price = sum(p.predicted_price * p.confidence for p in predictions) / total_weight
        avg_confidence = np.mean([p.confidence for p in predictions])
        
        ensemble_prediction = PricePrediction(
            item_name=predictions[0].item_name,
            current_price=predictions[0].current_price,
            predicted_price=weighted_price,
            confidence=avg_confidence,
            timeframe_hours=predictions[0].timeframe_hours,
            model_name="ensemble",
            features_used=predictions[0].features_used,
            prediction_date=datetime.now()
        )
        
        return ensemble_prediction

class MarketTrendAnalyzer:
    """Advanced market trend analysis using ML"""
    
    def __init__(self):
        self.config = FloatCheckerConfig()
        self.logger = logging.getLogger(__name__)
        self.trend_models = {}
    
    async def analyze_market_trends(self, market_data: Dict[str, List[Tuple[datetime, float]]], 
                                  trade_history: List[TradeRecord]) -> List[MarketTrend]:
        """Analyze market trends using ML techniques"""
        trends = []
        
        try:
            # Overall market trend
            overall_trend = await self._analyze_overall_trend(market_data, trade_history)
            if overall_trend:
                trends.append(overall_trend)
            
            # Category-specific trends
            category_trends = await self._analyze_category_trends(market_data, trade_history)
            trends.extend(category_trends)
            
            # Volatility trends
            volatility_trend = await self._analyze_volatility_trend(market_data)
            if volatility_trend:
                trends.append(volatility_trend)
            
        except Exception as e:
            self.logger.error(f"Error analyzing market trends: {e}")
        
        return trends
    
    async def _analyze_overall_trend(self, market_data: Dict[str, List[Tuple[datetime, float]]], 
                                   trade_history: List[TradeRecord]) -> Optional[MarketTrend]:
        """Analyze overall market trend"""
        try:
            # Combine all price data
            all_prices = []
            for item_prices in market_data.values():
                all_prices.extend([p[1] for p in item_prices])
            
            if len(all_prices) < 10:
                return None
            
            # Calculate trend using linear regression
            recent_prices = all_prices[-50:]  # Last 50 data points
            x = np.array(range(len(recent_prices)))
            y = np.array(recent_prices)
            
            slope, intercept = np.polyfit(x, y, 1)
            
            # Determine trend direction
            if slope > 0.1:
                direction = "bullish"
                strength = min(abs(slope) / np.mean(recent_prices), 1.0)
            elif slope < -0.1:
                direction = "bearish"
                strength = min(abs(slope) / np.mean(recent_prices), 1.0)
            else:
                direction = "sideways"
                strength = 0.5
            
            # Calculate confidence based on R²
            correlation = np.corrcoef(x, y)[0, 1]
            confidence = abs(correlation) if not np.isnan(correlation) else 0.5
            
            # Key factors analysis
            key_factors = []
            if len(trade_history) > 10:
                recent_trades = trade_history[-10:]
                profitable_trades = len([t for t in recent_trades if t.actual_profit and t.actual_profit > 0])
                if profitable_trades / len(recent_trades) > 0.7:
                    key_factors.append("high_success_rate")
                elif profitable_trades / len(recent_trades) < 0.3:
                    key_factors.append("low_success_rate")
            
            return MarketTrend(
                trend_direction=direction,
                strength=strength,
                confidence=confidence,
                duration_hours=24,
                key_factors=key_factors,
                analysis_date=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing overall trend: {e}")
            return None
    
    async def _analyze_category_trends(self, market_data: Dict[str, List[Tuple[datetime, float]]], 
                                     trade_history: List[TradeRecord]) -> List[MarketTrend]:
        """Analyze trends by weapon category"""
        category_trends = []
        
        try:
            # Group data by category
            categories = {}
            for item_name, prices in market_data.items():
                category = item_name.split(' | ')[0] if ' | ' in item_name else item_name.split(' ')[0]
                if category not in categories:
                    categories[category] = []
                categories[category].extend([p[1] for p in prices])
            
            # Analyze each category
            for category, prices in categories.items():
                if len(prices) < 5:
                    continue
                
                recent_prices = prices[-20:]  # Last 20 data points
                x = np.array(range(len(recent_prices)))
                y = np.array(recent_prices)
                
                slope, _ = np.polyfit(x, y, 1)
                
                if slope > 0.05:
                    direction = "bullish"
                    strength = min(abs(slope) / np.mean(recent_prices), 1.0)
                elif slope < -0.05:
                    direction = "bearish"
                    strength = min(abs(slope) / np.mean(recent_prices), 1.0)
                else:
                    direction = "sideways"
                    strength = 0.3
                
                correlation = np.corrcoef(x, y)[0, 1]
                confidence = abs(correlation) if not np.isnan(correlation) else 0.5
                
                trend = MarketTrend(
                    trend_direction=f"{category}_{direction}",
                    strength=strength,
                    confidence=confidence,
                    duration_hours=12,
                    key_factors=[f"category_{category}"],
                    analysis_date=datetime.now()
                )
                
                category_trends.append(trend)
        
        except Exception as e:
            self.logger.error(f"Error analyzing category trends: {e}")
        
        return category_trends
    
    async def _analyze_volatility_trend(self, market_data: Dict[str, List[Tuple[datetime, float]]]) -> Optional[MarketTrend]:
        """Analyze market volatility trends"""
        try:
            all_volatilities = []
            
            for prices in market_data.values():
                if len(prices) < 5:
                    continue
                
                price_values = [p[1] for p in prices]
                returns = [(price_values[i] - price_values[i-1]) / price_values[i-1] 
                          for i in range(1, len(price_values))]
                volatility = np.std(returns)
                all_volatilities.append(volatility)
            
            if not all_volatilities:
                return None
            
            avg_volatility = np.mean(all_volatilities)
            
            # Determine volatility level
            if avg_volatility > 0.05:
                direction = "high_volatility"
                strength = min(avg_volatility * 10, 1.0)
            elif avg_volatility < 0.02:
                direction = "low_volatility"
                strength = 0.8
            else:
                direction = "normal_volatility"
                strength = 0.5
            
            return MarketTrend(
                trend_direction=direction,
                strength=strength,
                confidence=0.8,
                duration_hours=6,
                key_factors=["volatility_analysis"],
                analysis_date=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility trend: {e}")
            return None

class AIDecisionEngine:
    """AI-powered automated decision making"""
    
    def __init__(self, predictor: MLPredictor, trend_analyzer: MarketTrendAnalyzer):
        self.predictor = predictor
        self.trend_analyzer = trend_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Decision thresholds
        self.decision_thresholds = {
            'buy_confidence_min': 0.7,
            'sell_confidence_min': 0.6,
            'min_profit_percentage': 15.0,
            'max_risk_percentage': 5.0,
            'trend_agreement_weight': 0.3
        }
    
    async def generate_trading_signals(self, opportunities: List[TradeOpportunity], 
                                     market_trends: List[MarketTrend],
                                     portfolio: List[PortfolioItem]) -> List[TradingSignal]:
        """Generate AI-powered trading signals"""
        signals = []
        
        for opportunity in opportunities:
            try:
                signal = await self._analyze_opportunity(opportunity, market_trends, portfolio)
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error analyzing opportunity {opportunity.item_name}: {e}")
        
        # Sort signals by strength and confidence
        signals.sort(key=lambda s: s.strength * s.confidence, reverse=True)
        
        return signals[:10]  # Return top 10 signals
    
    async def _analyze_opportunity(self, opportunity: TradeOpportunity, 
                                 market_trends: List[MarketTrend],
                                 portfolio: List[PortfolioItem]) -> Optional[TradingSignal]:
        """Analyze individual trading opportunity"""
        try:
            reasoning = []
            
            # Base signal from opportunity
            if opportunity.profit_percentage >= self.decision_thresholds['min_profit_percentage']:
                signal_type = "buy"
                base_strength = min(opportunity.profit_percentage / 50.0, 1.0)
                reasoning.append(f"High profit potential: {opportunity.profit_percentage:.1f}%")
            else:
                signal_type = "hold"
                base_strength = 0.3
                reasoning.append("Insufficient profit potential")
            
            # Market trend analysis
            relevant_trends = [t for t in market_trends if opportunity.item_name.split(' | ')[0] in t.trend_direction]
            
            trend_multiplier = 1.0
            if relevant_trends:
                trend = relevant_trends[0]
                if trend.trend_direction.endswith('bullish') and signal_type == "buy":
                    trend_multiplier = 1.0 + (trend.strength * self.decision_thresholds['trend_agreement_weight'])
                    reasoning.append(f"Bullish trend supports buy signal (strength: {trend.strength:.2f})")
                elif trend.trend_direction.endswith('bearish') and signal_type == "buy":
                    trend_multiplier = 1.0 - (trend.strength * self.decision_thresholds['trend_agreement_weight'])
                    reasoning.append(f"Bearish trend conflicts with buy signal")
            
            # Portfolio concentration check
            portfolio_value = sum(item.current_market_price for item in portfolio)
            concentration_risk = opportunity.market_price / max(portfolio_value, 1.0)
            
            if concentration_risk > self.decision_thresholds['max_risk_percentage'] / 100:
                base_strength *= 0.7
                reasoning.append("High concentration risk reduces signal strength")
            
            # Risk assessment
            risk_multiplier = 1.0
            if opportunity.risk_level.value == "high":
                risk_multiplier = 0.7
                reasoning.append("High risk level detected")
            elif opportunity.risk_level.value == "critical":
                signal_type = "hold"
                base_strength = 0.1
                reasoning.append("Critical risk - avoid trade")
            
            # Calculate final strength and confidence
            final_strength = base_strength * trend_multiplier * risk_multiplier
            final_strength = min(max(final_strength, 0.0), 1.0)
            
            # Confidence based on multiple factors
            confidence_factors = [
                opportunity.confidence_score / 100.0,
                min(opportunity.profit_percentage / 30.0, 1.0),
                1.0 - concentration_risk * 10,  # Penalize high concentration
            ]
            
            if relevant_trends:
                confidence_factors.append(relevant_trends[0].confidence)
            
            final_confidence = np.mean(confidence_factors)
            final_confidence = min(max(final_confidence, 0.0), 1.0)
            
            # Set price targets
            if signal_type == "buy":
                price_target = opportunity.target_price
                stop_loss = opportunity.market_price * 0.9  # 10% stop loss
            else:
                price_target = opportunity.market_price
                stop_loss = opportunity.market_price * 0.95  # 5% stop loss
            
            # Only generate signal if confidence meets threshold
            min_confidence = self.decision_thresholds.get(f"{signal_type}_confidence_min", 0.5)
            if final_confidence < min_confidence:
                signal_type = "hold"
                reasoning.append(f"Confidence {final_confidence:.2f} below threshold {min_confidence:.2f}")
            
            signal = TradingSignal(
                item_name=opportunity.item_name,
                signal_type=signal_type,
                strength=final_strength,
                confidence=final_confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                reasoning=reasoning,
                generated_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=6)
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error in opportunity analysis: {e}")
            return None

# Test function
async def test_ml_engine():
    """Test ML engine functionality"""
    print("🧪 Testing ML Engine...")
    
    # Test feature engineering
    print("Test 1: Feature Engineering")
    feature_engine = FeatureEngineering()
    
    mock_price_history = [(datetime.now() - timedelta(days=i), 100.0 + i) for i in range(10)]
    features = feature_engine.extract_price_features(mock_price_history)
    print(f"✅ Extracted {len(features)} price features")
    
    item_features = feature_engine.extract_item_features("AK-47 | Redline (Factory New)")
    print(f"✅ Extracted {len(item_features)} item features")
    
    # Test ML predictor
    print("Test 2: ML Predictor")
    predictor = MLPredictor()
    print(f"✅ Initialized {len(predictor.models)} ML models")
    
    # Test trend analyzer
    print("Test 3: Market Trend Analyzer")
    trend_analyzer = MarketTrendAnalyzer()
    mock_market_data = {
        "AK-47 | Redline": [(datetime.now() - timedelta(hours=i), 50.0 + i) for i in range(24)]
    }
    trends = await trend_analyzer.analyze_market_trends(mock_market_data, [])
    print(f"✅ Analyzed {len(trends)} market trends")
    
    # Test AI decision engine
    print("Test 4: AI Decision Engine")
    ai_engine = AIDecisionEngine(predictor, trend_analyzer)
    print("✅ AI Decision Engine initialized")
    
    print("✅ ML Engine test completed successfully")

if __name__ == "__main__":
    asyncio.run(test_ml_engine())