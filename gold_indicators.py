"""
Gold Trading Indicators - Buy/Sell Signals Based on Supply & Demand Levels
Python 3.14+ Trading System
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import ta  # Technical Analysis library


class GoldIndicators:
    """Calculate trading indicators for gold with supply/demand analysis"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        self.signals = pd.DataFrame(index=data.index)
        
    def identify_supply_demand_levels(self, lookback: int = 20) -> Tuple[List[float], List[float]]:
        """
        Identify supply and demand levels using swing highs/lows
        
        Args:
            lookback: Number of periods to look back for swing identification
            
        Returns:
            Tuple of (supply_levels, demand_levels)
        """
        high = self.data['high'].values
        low = self.data['low'].values
        
        supply_levels = []
        demand_levels = []
        
        for i in range(lookback, len(high) - lookback):
            # Supply: local maxima
            if high[i] == max(high[i-lookback:i+lookback+1]):
                supply_levels.append(high[i])
            
            # Demand: local minima
            if low[i] == min(low[i-lookback:i+lookback+1]):
                demand_levels.append(low[i])
        
        # Remove duplicates and sort
        supply_levels = sorted(list(set([round(x, 2) for x in supply_levels])), reverse=True)
        demand_levels = sorted(list(set([round(x, 2) for x in demand_levels])))
        
        return supply_levels[:5], demand_levels[:5]  # Return top 5 levels
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        return ta.momentum.rsi(self.data['close'], window=period)
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicators"""
        macd_line = ta.trend.macd(self.data['close'], window_fast=fast, window_slow=slow)
        # macd returns 3 series: MACD line, Signal line, Histogram
        # We need to calculate signal line separately
        macd_vals = ta.trend.macd(self.data['close'], window_fast=fast, window_slow=slow)
        # Return MACD, Signal (EMA of MACD), and Histogram
        return macd_vals, macd_vals.ewm(span=signal).mean(), macd_vals - macd_vals.ewm(span=signal).mean()
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        # Calculate middle band (SMA)
        middle = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def identify_breakout_levels(self, period: int = 10) -> Tuple[float, float]:
        """
        Identify breakout resistance and support levels
        
        Returns:
            Tuple of (resistance, support)
        """
        recent_high = self.data['high'].tail(period).max()
        recent_low = self.data['low'].tail(period).min()
        return recent_high, recent_low
    
    def calculate_buy_signals(self) -> pd.Series:
        """
        Generate BUY signals based on multiple indicators
        
        Conditions:
        - Price touches or bounces from demand level
        - RSI crosses above oversold (30)
        - MACD crossover bullish
        - Price above lower Bollinger Band
        """
        rsi = self.calculate_rsi()
        macd, macd_signal, _ = self.calculate_macd()
        _, _, bb_lower = self.calculate_bollinger_bands()
        
        supply, demand = self.identify_supply_demand_levels()
        demand_level = demand[0] if demand else self.data['low'].min()
        
        buy_signals = pd.Series(0, index=self.data.index)
        
        for i in range(1, len(self.data)):
            close_price = self.data['close'].iloc[i]
            prev_close = self.data['close'].iloc[i-1]
            
            # Condition 1: Price near demand level
            near_demand = (close_price >= demand_level * 0.98 and 
                          close_price <= demand_level * 1.02)
            
            # Condition 2: RSI oversold bounce
            rsi_bounce = (rsi.iloc[i-1] < 30 and rsi.iloc[i] > 30)
            
            # Condition 3: MACD bullish crossover
            macd_crossover = (macd.iloc[i-1] < macd_signal.iloc[i-1] and 
                             macd.iloc[i] > macd_signal.iloc[i])
            
            # Condition 4: Price above lower BB
            above_bb = close_price > bb_lower.iloc[i]
            
            if (near_demand or rsi_bounce or macd_crossover) and above_bb:
                buy_signals.iloc[i] = 1
        
        return buy_signals
    
    def calculate_sell_signals(self) -> pd.Series:
        """
        Generate SELL signals based on multiple indicators
        
        Conditions:
        - Price reaches or rejects from supply level
        - RSI crosses below overbought (70)
        - MACD crossover bearish
        - Price below upper Bollinger Band
        """
        rsi = self.calculate_rsi()
        macd, macd_signal, _ = self.calculate_macd()
        bb_upper, _, _ = self.calculate_bollinger_bands()
        
        supply, _ = self.identify_supply_demand_levels()
        supply_level = supply[0] if supply else self.data['high'].max()
        
        sell_signals = pd.Series(0, index=self.data.index)
        
        for i in range(1, len(self.data)):
            close_price = self.data['close'].iloc[i]
            
            # Condition 1: Price near supply level
            near_supply = (close_price >= supply_level * 0.98 and 
                          close_price <= supply_level * 1.02)
            
            # Condition 2: RSI overbought rejection
            rsi_rejection = (rsi.iloc[i-1] > 70 and rsi.iloc[i] < 70)
            
            # Condition 3: MACD bearish crossover
            macd_crossover = (macd.iloc[i-1] > macd_signal.iloc[i-1] and 
                             macd.iloc[i] < macd_signal.iloc[i])
            
            # Condition 4: Price below upper BB
            below_bb = close_price < bb_upper.iloc[i]
            
            if (near_supply or rsi_rejection or macd_crossover) and below_bb:
                sell_signals.iloc[i] = -1
        
        return sell_signals
    
    def calculate_position_size(self, account_size: float, risk_percent: float = 2.0,
                               entry: float = 0, stop_loss: float = 0) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            account_size: Total trading account size
            risk_percent: Risk percentage per trade (default 2%)
            entry: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Number of units to trade
        """
        if entry == 0 or stop_loss == 0:
            return 0
        
        risk_amount = account_size * (risk_percent / 100)
        price_risk = abs(entry - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return round(position_size, 2)
    
    def generate_trading_report(self) -> Dict:
        """Generate comprehensive trading analysis report"""
        supply, demand = self.identify_supply_demand_levels()
        buy_sigs = self.calculate_buy_signals()
        sell_sigs = self.calculate_sell_signals()
        resistance, support = self.identify_breakout_levels()
        
        rsi = self.calculate_rsi()
        macd, macd_signal, macd_hist = self.calculate_macd()
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands()
        
        current_price = self.data['close'].iloc[-1]
        
        report = {
            'current_price': round(current_price, 2),
            'supply_levels': [round(x, 2) for x in supply],
            'demand_levels': [round(x, 2) for x in demand],
            'resistance': round(resistance, 2),
            'support': round(support, 2),
            'rsi': round(rsi.iloc[-1], 2),
            'macd': round(macd.iloc[-1], 4),
            'macd_signal': round(macd_signal.iloc[-1], 4),
            'bb_upper': round(bb_upper.iloc[-1], 2),
            'bb_middle': round(bb_middle.iloc[-1], 2),
            'bb_lower': round(bb_lower.iloc[-1], 2),
            'buy_signals_count': int(buy_sigs.sum()),
            'sell_signals_count': int(abs(sell_sigs.sum())),
            'last_signal': 'BUY' if buy_sigs.iloc[-1] == 1 else ('SELL' if sell_sigs.iloc[-1] == -1 else 'NONE'),
            'trend': 'BULLISH' if macd.iloc[-1] > macd_signal.iloc[-1] else 'BEARISH'
        }
        
        return report


if __name__ == "__main__":
    print("Gold Trading Indicators Module Loaded")
    print("Use: from gold_indicators import GoldIndicators")
