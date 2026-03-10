def scan(candles, symbol):
    if len(candles) < 20:
        return None
    closes = [c['close'] for c in candles]
    highs = [c['high'] for c in candles]
    sma5 = sum(closes[-5:]) / 5
    sma20 = sum(closes[-20:]) / 20
    atr = candles[-1].get('atr', candles[-1]['high'] - candles[-1]['low'])
    current = closes[-1]
    if current > max(highs[-5:-1]) and current > sma5 > sma20:
        return {
            'signal': 'long',
            'entry': current,
            'stop': current - atr * 2.0,
            'target': current + atr * 4.0,
            'reason': 'momentum_breakout'
        }
    return None