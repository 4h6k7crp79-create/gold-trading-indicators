#!/usr/bin/env python3
"""
Gold Trading Indicator - Main Standalone Runner
Python 3.14+ Compatible

Fetches real gold price data and generates buy/sell signals
"""

import sys
from typing import Dict
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from tradelocker_gold_indicator import GoldTradingIndicator


def fetch_gold_data(period: str = "1mo", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch gold price data from Yahoo Finance
    
    Args:
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, etc.)
        interval: Data interval (1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
        
    Returns:
        DataFrame with OHLCV data
    """
    if not HAS_YFINANCE:
        print("ERROR: yfinance not installed. Install with: pip install yfinance")
        sys.exit(1)
    
    print(f"📊 Fetching gold (XAUUSD) data for {period} with {interval} interval...")
    
    try:
        # GC=F is gold futures on Yahoo Finance
        gold_data = yf.download("GC=F", period=period, interval=interval, progress=False)
        
        if gold_data.empty:
            print("ERROR: Could not fetch gold data from Yahoo Finance")
            sys.exit(1)
        
        # Rename columns to lowercase
        gold_data.columns = [col.lower() for col in gold_data.columns]
        
        print(f"✅ Successfully fetched {len(gold_data)} candles")
        print(f"   Date range: {gold_data.index[0]} to {gold_data.index[-1]}")
        
        return gold_data
    
    except Exception as e:
        print(f"ERROR fetching data: {e}")
        sys.exit(1)


def generate_sample_data() -> Dict:
    """
    Generate sample gold price data for testing
    Returns Dict with OHLCV arrays
    """
    print("📊 Generating sample gold price data for testing...")
    
    np.random.seed(42)
    base_price = 1950.0
    prices = [base_price]
    
    for _ in range(99):
        change = np.random.uniform(-5, 5)
        prices.append(prices[-1] + change)
    
    close_prices = np.array(prices)
    
    bars = {
        'open': close_prices * (1 + np.random.uniform(-0.002, 0.002, len(close_prices))),
        'high': close_prices + np.abs(np.random.uniform(0, 5, len(close_prices))),
        'low': close_prices - np.abs(np.random.uniform(0, 5, len(close_prices))),
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, len(close_prices))
    }
    
    return bars


def print_signal_emoji(signal_type: str) -> str:
    """Return emoji for signal type"""
    if signal_type == "BUY":
        return "🟢"
    elif signal_type == "SELL":
        return "🔴"
    else:
        return "⚪"


def analyze_gold_trading(use_sample: bool = False):
    """
    Main analysis function
    """
    print("\n" + "="*70)
    print("🏆 GOLD TRADING INDICATOR v1.0 - Python 3.14+")
    print("="*70)
    
    # Get data
    if use_sample:
        bars = generate_sample_data()
    else:
        gold_data = fetch_gold_data(period="2mo", interval="1h")
        bars = {
            'open': gold_data['open'].values,
            'high': gold_data['high'].values,
            'low': gold_data['low'].values,
            'close': gold_data['close'].values,
            'volume': gold_data['volume'].values
        }
    
    print(f"\n📈 Analyzing {len(bars['close'])} candles...\n")
    
    # Initialize and calculate
    indicator = GoldTradingIndicator()
    result = indicator.calculate(bars)
    
    # Determine signal
    if result['buy_signal'] == 1:
        signal = "BUY"
    elif result['sell_signal'] == 1:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    signal_emoji = print_signal_emoji(signal)
    
    # Print results
    print(f"{signal_emoji} SIGNAL: {signal}")
    print(f"   Strength: {result['signal_strength']}/100 {'⭐' * int(result['signal_strength']/20)}")
    
    print(f"\n💰 PRICE INFO")
    print(f"   Current Price: ${result['current_price']}")
    print(f"   Resistance: ${result['resistance']}")
    print(f"   Support: ${result['support']}")
    print(f"   Range: ${result['support']} - ${result['resistance']}")
    
    print(f"\n📊 SUPPLY & DEMAND LEVELS")
    if result['supply_levels']:
        print(f"   Supply (Resistance): {[f'${x}' for x in result['supply_levels']]}")
    if result['demand_levels']:
        print(f"   Demand (Support): {[f'${x}' for x in result['demand_levels']]}")
    
    print(f"\n📈 TECHNICAL INDICATORS")
    print(f"   RSI (14): {result['rsi']}")
    print(f"      Status: {result['rsi_status']}")
    if result['rsi'] < 30:
        print(f"      ↳ OVERSOLD - Potential bounce expected")
    elif result['rsi'] > 70:
        print(f"      ↳ OVERBOUGHT - Potential pullback expected")
    
    print(f"\n   MACD:")
    print(f"      Line: {result['macd']}")
    print(f"      Signal: {result['signal_line']}")
    if result['macd'] > result['signal_line']:
        print(f"      ↳ BULLISH crossover")
    else:
        print(f"      ↳ BEARISH crossover")
    
    print(f"\n   Bollinger Bands:")
    print(f"      Upper: ${result['bb_upper']}")
    print(f"      Middle: ${result['bb_middle']}")
    print(f"      Lower: ${result['bb_lower']}")
    
    position = "Above Upper (Overbought)" if result['current_price'] > result['bb_upper'] else \
               "Below Lower (Oversold)" if result['current_price'] < result['bb_lower'] else \
               "Within Bands"
    print(f"      Position: {position}")
    
    print(f"\n📊 TREND ANALYSIS")
    trend_arrow = "📈" if result['trend'] == "BULLISH" else "📉"
    print(f"   {trend_arrow} Trend: {result['trend']}")
    
    print(f"\n💡 TRADING RECOMMENDATIONS")
    if signal == "BUY":
        print(f"   ✅ LONG ENTRY at ${result['current_price']}")
        if result['demand_levels']:
            print(f"   📍 Stop Loss: Below ${result['demand_levels'][0]}")
        if result['supply_levels']:
            print(f"   🎯 Take Profit: ${result['supply_levels'][0]}")
    elif signal == "SELL":
        print(f"   ❌ SHORT ENTRY at ${result['current_price']}")
        if result['supply_levels']:
            print(f"   📍 Stop Loss: Above ${result['supply_levels'][0]}")
        if result['demand_levels']:
            print(f"   🎯 Take Profit: ${result['demand_levels'][0]}")
    else:
        print(f"   ⏸️  No confirmed signal. Wait for better entry.")
        print(f"   📌 Watch these levels for entries:")
        if result['demand_levels']:
            print(f"      Support: ${result['demand_levels'][0]}")
        if result['supply_levels']:
            print(f"      Resistance: ${result['supply_levels'][0]}")
    
    print(f"\n" + "="*70)
    print("⚠️  DISCLAIMER: For educational purposes only. Trade at your own risk.")
    print("="*70 + "\n")


def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gold Trading Indicator - Real-time buy/sell signals'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample data instead of fetching live data'
    )
    
    args = parser.parse_args()
    
    try:
        analyze_gold_trading(use_sample=args.sample)
    except KeyboardInterrupt:
        print("\n\n⏸️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
