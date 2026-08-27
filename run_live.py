#!/usr/bin/env python3
"""
Gold Trading Indicator - Live Data Runner
Fetches real gold prices and shows buy/sell signals
"""

import sys
import pandas as pd
import yfinance as yf
from gold_indicators import GoldIndicators

def fetch_gold_data():
    """Fetch live gold data from Yahoo Finance"""
    print("📊 Fetching live gold prices...")
    try:
        gold = yf.download("GC=F", period="2mo", interval="1h", progress=False)
        print(f"✅ Fetched {len(gold)} candles\n")
        
        # Ensure columns are lowercase
        gold.columns = [col.lower() for col in gold.columns]
        return gold
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def analyze():
    """Run the analysis"""
    data = fetch_gold_data()
    if data is None or data.empty:
        print("❌ No data available")
        return
    
    try:
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        # Initialize indicator with data
        indicator = GoldIndicators(data)
        print("✅ Indicator initialized")
        
        # Generate report
        report = indicator.generate_trading_report()
        print("✅ Report generated")
        
        # Print results
        print("\n" + "=" * 60)
        print("🏆 GOLD TRADING ANALYSIS")
        print("=" * 60)
        print(f"Current Price: ${report['current_price']}")
        print(f"Resistance: ${report['resistance']}")
        print(f"Support: ${report['support']}")
        
        print(f"\n📈 Supply & Demand Levels:")
        if report['supply_levels']:
            print(f"   Supply: {report['supply_levels']}")
        if report['demand_levels']:
            print(f"   Demand: {report['demand_levels']}")
        
        print(f"\n📊 Technical Indicators:")
        print(f"   RSI: {report['rsi']}")
        print(f"   MACD: {report['macd']}")
        print(f"   Trend: {report['trend']}")
        
        print(f"\n💡 Latest Signal: {report['last_signal']}")
        print(f"   Buy Signals (Total): {report['buy_signals_count']}")
        print(f"   Sell Signals (Total): {report['sell_signals_count']}")
        
        print("=" * 60)
        print("✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        analyze()
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
