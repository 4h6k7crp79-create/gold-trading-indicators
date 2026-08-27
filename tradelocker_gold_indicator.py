"""
Gold Trading Indicator for TradeLocker
Real-time Buy/Sell signals with Supply & Demand levels
Compatible with TradeLocker's Python indicator API
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class GoldTradingIndicator:
    """
    TradeLocker-compatible Gold Trading Indicator
    Provides buy/sell signals based on supply/demand levels and technical analysis
    """
    
    def __init__(self):
        self.name = "Gold Buy/Sell Indicator"
        self.version = "1.0"
        
    def calculate(self, bars: Dict) -> Dict:
        """
        Main calculation function called by TradeLocker
        
        Args:
            bars: Dictionary with OHLCV data from TradeLocker
                 {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        Returns:
            Dictionary with buy/sell signals and levels
        """
        
        close = np.array(bars['close'])
        high = np.array(bars['high'])
        low = np.array(bars['low'])
        
        if len(close) < 20:
            return self._empty_result()
        
        # Calculate indicators
        current_price = close[-1]
        rsi = self._calculate_rsi(close, 14)
        macd_line, signal_line, histogram = self._calculate_macd(close)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, 20, 2.0)
        
        # Supply and Demand levels
        supply_levels, demand_levels = self._identify_levels(high, low, 20)
        
        # Generate signals
        buy_signal = self._generate_buy_signal(
            current_price, rsi[-1], macd_line[-1], signal_line[-1],
            bb_lower[-1], demand_levels
        )
        
        sell_signal = self._generate_sell_signal(
            current_price, rsi[-1], macd_line[-1], signal_line[-1],
            bb_upper[-1], supply_levels
        )
        
        # Calculate support and resistance
        resistance = high[-20:].max()
        support = low[-20:].min()
        
        return {
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'current_price': round(current_price, 2),
            'rsi': round(rsi[-1], 2),
            'macd': round(macd_line[-1], 4),
            'signal_line': round(signal_line[-1], 4),
            'bb_upper': round(bb_upper[-1], 2),
            'bb_middle': round(bb_middle[-1], 2),
            'bb_lower': round(bb_lower[-1], 2),
            'resistance': round(resistance, 2),
            'support': round(support, 2),
            'supply_levels': [round(x, 2) for x in supply_levels[:3]],
            'demand_levels': [round(x, 2) for x in demand_levels[:3]],
            'trend': 'BULLISH' if macd_line[-1] > signal_line[-1] else 'BEARISH',
            'rsi_status': self._get_rsi_status(rsi[-1]),
            'signal_strength': self._calculate_signal_strength(buy_signal, sell_signal, rsi[-1])
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = np.zeros_like(prices)
        rs[:period] = 100. - 100. / (1. + up / down) if down != 0 else 0
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs[i] = 100. - 100. / (1. + up / down) if down != 0 else 0
        
        return rs
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.zeros_like(prices)
        multiplier = 2 / (period + 1)
        
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = prices[i] * multiplier + ema[i-1] * (1 - multiplier)
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands"""
        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        sma = np.pad(sma, (period-1, 0), mode='edge')
        
        std = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
        
        bb_upper = sma + (std * std_dev)
        bb_middle = sma
        bb_lower = sma - (std * std_dev)
        
        return bb_upper, bb_middle, bb_lower
    
    def _identify_levels(self, high: np.ndarray, low: np.ndarray, lookback: int = 20) -> Tuple[List[float], List[float]]:
        """Identify supply (resistance) and demand (support) levels"""
        supply_levels = []
        demand_levels = []
        
        for i in range(lookback, len(high) - lookback):
            # Supply: local maxima
            if high[i] == np.max(high[i-lookback:i+lookback+1]):
                supply_levels.append(high[i])
            
            # Demand: local minima
            if low[i] == np.min(low[i-lookback:i+lookback+1]):
                demand_levels.append(low[i])
        
        supply_levels = sorted(list(set([round(x, 2) for x in supply_levels])), reverse=True)
        demand_levels = sorted(list(set([round(x, 2) for x in demand_levels])))
        
        return supply_levels[:5], demand_levels[:5]
    
    def _generate_buy_signal(self, price: float, rsi: float, macd: float, signal: float, bb_lower: float, demand_levels: List[float]) -> int:
        """
        Generate buy signal (1 for buy, 0 for no signal)
        
        Buy conditions:
        - Price near demand level
        - RSI oversold or bouncing from oversold
        - MACD bullish crossover
        - Price above lower Bollinger Band
        """
        
        demand_level = demand_levels[0] if demand_levels else bb_lower
        near_demand = (price >= demand_level * 0.98) and (price <= demand_level * 1.02)
        
        rsi_bullish = rsi < 30 or (rsi > 35 and rsi < 50)
        macd_bullish = macd > signal
        bb_support = price > bb_lower
        
        if near_demand and rsi_bullish and macd_bullish and bb_support:
            return 1
        
        return 0
    
    def _generate_sell_signal(self, price: float, rsi: float, macd: float, signal: float, bb_upper: float, supply_levels: List[float]) -> int:
        """
        Generate sell signal (1 for sell, 0 for no signal)
        
        Sell conditions:
        - Price near supply level
        - RSI overbought
        - MACD bearish crossover
        - Price below upper Bollinger Band
        """
        
        supply_level = supply_levels[0] if supply_levels else bb_upper
        near_supply = (price >= supply_level * 0.98) and (price <= supply_level * 1.02)
        
        rsi_bearish = rsi > 70 or (rsi < 65 and rsi > 50)
        macd_bearish = macd < signal
        bb_resistance = price < bb_upper
        
        if near_supply and rsi_bearish and macd_bearish and bb_resistance:
            return 1
        
        return 0
    
    def _get_rsi_status(self, rsi: float) -> str:
        """Get RSI status description"""
        if rsi < 30:
            return "OVERSOLD"
        elif rsi > 70:
            return "OVERBOUGHT"
        elif rsi > 50:
            return "BULLISH"
        elif rsi < 50:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_signal_strength(self, buy: int, sell: int, rsi: float) -> float:
        """Calculate signal strength (0-100)"""
        strength = 0
        
        if buy == 1:
            strength = abs(50 - rsi) * 2  # RSI deviation from midpoint
            strength = min(strength, 100)
        elif sell == 1:
            strength = abs(50 - rsi) * 2
            strength = min(strength, 100)
        
        return round(strength, 2)
    
    def _empty_result(self) -> Dict:
        """Return empty result when insufficient data"""
        return {
            'buy_signal': 0,
            'sell_signal': 0,
            'current_price': 0,
            'rsi': 0,
            'macd': 0,
            'signal_line': 0,
            'bb_upper': 0,
            'bb_middle': 0,
            'bb_lower': 0,
            'resistance': 0,
            'support': 0,
            'supply_levels': [],
            'demand_levels': [],
            'trend': 'NEUTRAL',
            'rsi_status': 'NEUTRAL',
            'signal_strength': 0
        }


# TradeLocker Integration Function
def on_bar(bars: Dict) -> Dict:
    """
    TradeLocker callback function
    Called on each new bar/candle
    
    Args:
        bars: OHLCV data from TradeLocker
    
    Returns:
        Signal data to display on chart
    """
    indicator = GoldTradingIndicator()
    result = indicator.calculate(bars)
    
    # Format for TradeLocker display
    return {
        'buy_signal': result['buy_signal'],
        'sell_signal': result['sell_signal'],
        'current_price': result['current_price'],
        'rsi': result['rsi'],
        'support': result['support'],
        'resistance': result['resistance'],
        'trend': result['trend'],
        'signal_strength': result['signal_strength']
    }


if __name__ == "__main__":
    print("Gold Trading Indicator for TradeLocker")
    print("Version: 1.0")
    print("Ready to use with TradeLocker API")
