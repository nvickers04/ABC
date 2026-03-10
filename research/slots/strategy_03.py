def scan(candles, symbol):
    if len(candles) < 15:
        return None
    closes = [c['close'] for c in candles]
    if closes[-1] > closes[-2] > closes[-3] and candles[-1]['volume'] > candles[-2].get('volume', 0) * 1.2:
        return {
            'signal': 'bull_call_vertical',
            'entry': candles[-1]['close'],
            'reason': 'trend_in_high_vol',
            'legs': 'long lower strike, short higher strike'
        }
    return None