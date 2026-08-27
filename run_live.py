#!/usr/bin/env python3
"""
Gold Trading Indicator - Live Data Runner
Fetches real gold prices and shows buy/sell signals
"""

import sys
import yfinance as yf
from gold_indicators import GoldIndicators

def fetch_gold_data():
    """Fetch live gold data from Yahoo Finance"""
    print("📊 Fetching live gold prices...")
    try:
        gold = yf.download("GC=F", period="2mo", interval="1h", progress=False)
        print(f"✅ Fetched {len(gold)} candles\n")
        return gold
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def analyze():
    """Run the analysis"""
    data = fetch_gold_data()
    if data is None:
        return
    
    # Prepare data for indicator
    bars = {
        'open': data['Open'].values,
        'high': data['High'].values,
        'low': data['Low'].values,
        'close': data['Close'].values,
        'volume': data['Volume'].values
    }
    
    # Run indicator
    indicator = GoldIndicators()
    result = indicator.analyze(bars)
    
    # Print results
    print("=" * 60)
    print("🏆 GOLD TRADING ANALYSIS")
    print("=" * 60)
    print(f"Current Price: ${bars['close'][-1]:.2f}")
    print(f"High (2mo): ${bars['high'].max():.2f}")
    print(f"Low (2mo): ${bars['low'].min():.2f}")
    print(f"\nSignal: {result.get('signal', 'HOLD')}")
    print(f"Strength: {result.get('strength', 'N/A')}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        analyze()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
