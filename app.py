from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from binance.client import Client
import base64
import io
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Replace with your Binance API key and secret
BINANCE_API_KEY = 'your_api_key'
BINANCE_API_SECRET = 'your_api_secret'

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_symbols():
    info = client.get_exchange_info()
    symbols = [s['symbol'] for s in info['symbols']]
    symbols.sort()
    return symbols

def get_historical_data(symbol, interval):
    klines = client.get_klines(symbol=symbol, interval=interval)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    # Convert columns to numeric types
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')
    print('Historical data:', data.head())  # Debugging print
    return data[['open', 'high', 'low', 'close', 'volume']]

def generate_technical_analysis(data):
    data['SMA20'] = data['close'].rolling(window=20).mean()
    data['SMA50'] = data['close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['close'])
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = compute_macd(data['close'])
    return data

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def generate_signal_and_levels(data, interval):
    latest_close = data['close'].iloc[-1]
    sma20 = data['SMA20'].iloc[-1]
    sma50 = data['SMA50'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    macd = data['MACD'].iloc[-1]
    macd_signal = data['MACD_signal'].iloc[-1]

    print(f"Debug info: latest_close={latest_close}, sma20={sma20}, sma50={sma50}, rsi={rsi}, macd={macd}, macd_signal={macd_signal}")

    signal = None
    tp = None
    sl = None
    tp_percentage = 0
    sl_percentage = 0
    estimated_tp_time = None
    tp_probability = 0

    # Simple strategy: Buy when SMA20 > SMA50 and RSI < 70 and MACD > MACD Signal
    if sma20 > sma50 and rsi < 70 and macd > macd_signal:
        signal = 'BUY'
        tp = latest_close + (latest_close - data['low'].iloc[-14:].min()) * 2  # Dynamic TP based on recent low
        sl = latest_close - (data['high'].iloc[-14:].max() - latest_close) * 2  # Dynamic SL based on recent high
    elif sma20 < sma50 and rsi > 30 and macd < macd_signal:
        signal = 'SELL'
        tp = latest_close - (data['high'].iloc[-14:].max() - latest_close) * 2  # Dynamic TP based on recent high
        sl = latest_close + (latest_close - data['low'].iloc[-14:].min()) * 2  # Dynamic SL based on recent low

    if tp is not None and sl is not None:
        tp_percentage = (tp - latest_close) / latest_close * 100 if signal == 'BUY' else (latest_close - tp) / latest_close * 100
        sl_percentage = (sl - latest_close) / latest_close * 100 if signal == 'BUY' else (latest_close - sl) / latest_close * 100

        # Estimate time to reach TP based on recent volatility
        recent_volatility = data['close'].pct_change().rolling(window=10).std().iloc[-1]
        if recent_volatility > 0:
            estimated_tp_time_units = abs((tp - latest_close) / (recent_volatility * latest_close))
            interval_mapping = {
                '1m': timedelta(minutes=1),
                '5m': timedelta(minutes=5),
                '15m': timedelta(minutes=15),
                '30m': timedelta(minutes=30),
                '1h': timedelta(hours=1),
                '2h': timedelta(hours=2),
                '4h': timedelta(hours=4),
                '6h': timedelta(hours=6),
                '8h': timedelta(hours=8),
                '12h': timedelta(hours=12),
                '1d': timedelta(days=1),
                '3d': timedelta(days=3),
                '1w': timedelta(weeks=1),
                '1M': timedelta(days=30)  # Approximation for a month
            }
            estimated_tp_time = datetime.now() + estimated_tp_time_units * interval_mapping[interval]

        # Estimate TP probability based on historical data
        tp_probability = estimate_tp_probability(data, signal, tp, sl)

    print('Signal:', signal, 'TP:', tp, 'SL:', sl, 'TP %:', tp_percentage, 'SL %:', sl_percentage, 'Estimated TP Time:', estimated_tp_time, 'TP Probability:', tp_probability)  # Debugging print

    return signal, tp, sl, tp_percentage, sl_percentage, estimated_tp_time, tp_probability

def estimate_tp_probability(data, signal, tp, sl):
    successful_trades = 0
    total_trades = 0
    
    for i in range(len(data) - 1):
        if signal == 'BUY':
            if data['low'].iloc[i+1] <= sl:
                # Stop loss hit before next candle
                continue
            if data['high'].iloc[i+1] >= tp:
                successful_trades += 1
            total_trades += 1
        elif signal == 'SELL':
            if data['high'].iloc[i+1] >= sl:
                # Stop loss hit before next candle
                continue
            if data['low'].iloc[i+1] <= tp:
                successful_trades += 1
            total_trades += 1
    
    probability = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
    return probability

def get_logs():
    # Example logs
    return ["Log entry 1", "Log entry 2", "Log entry 3"]

def get_status():
    # Example status
    return "Running"

def get_top_signals(interval):
    symbols = get_symbols()
    signals = []

    for symbol in symbols:
        data = get_historical_data(symbol, interval)
        analysis_data = generate_technical_analysis(data)
        signal, tp, sl, tp_percentage, sl_percentage, estimated_tp_time, tp_probability = generate_signal_and_levels(analysis_data, interval)
        
        if signal is not None:
            signals.append({
                'symbol': symbol,
                'signal': signal,
                'tp': tp,
                'sl': sl,
                'tp_percentage': tp_percentage,
                'sl_percentage': sl_percentage,
                'estimated_tp_time': estimated_tp_time,
                'tp_probability': tp_probability
            })

    signals.sort(key=lambda x: x['tp_probability'], reverse=True)
    return signals[:10]

@app.route('/')
def home():
    symbols = get_symbols()
    return render_template('index.html', symbols=symbols)

@app.route('/logs')
def logs():
    logs = get_logs()
    return render_template('logs.html', logs=logs)

@app.route('/dashboard')
def dashboard():
    status = get_status()
    return render_template('dashboard.html', status=status)

@app.route('/connect')
def connect():
    status = get_status()
    return render_template('connect.html', status=status)

@app.route('/top-signals')
def top_signals():
    interval = '1h'  # Example interval
    signals = get_top_signals(interval)
    return render_template('top_signals.html', signals=signals, interval=interval)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data['symbol']
    interval = data['interval']
    
    print('Received request:', data)  # Debugging print

    historical_data = get_historical_data(symbol, interval)
    analysis_data = generate_technical_analysis(historical_data)
    signal, tp, sl, tp_percentage, sl_percentage, estimated_tp_time, tp_probability = generate_signal_and_levels(analysis_data, interval)
    
    fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    mpf.plot(analysis_data, type='candle', ax=ax, mav=(20, 50), volume=ax2)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)  # Close the figure to free memory

    if signal is not None and tp is not None and sl is not None:
        commentary = (f"Sygnał: {signal}.\n"
                      f"Take Profit: {tp:.2f} ({tp_percentage:.2f}%).\n"
                      f"Stop Loss: {sl:.2f} ({sl_percentage:.2f}%).\n"
                      f"Procent Zysku: {tp_percentage:.2f}%.\n"
                      f"Procent Straty: {sl_percentage:.2f}%.\n"
                      f"Przewidywany czas osiągnięcia TP: {estimated_tp_time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                      f"Szansa osiągnięcia TP: {tp_probability:.2f}%")
    else:
        commentary = "Brak sygnału."

    print('Response:', {'image': plot_url, 'commentary': commentary, 'signal': signal, 'tp': tp, 'sl': sl, 'tp_percentage': tp_percentage, 'sl_percentage': sl_percentage, 'estimated_tp_time': estimated_tp_time, 'tp_probability': tp_probability})  # Debugging print

    return jsonify({'image': plot_url, 'commentary': commentary, 'signal': signal, 'tp': tp, 'sl': sl, 'tp_percentage': tp_percentage, 'sl_percentage': sl_percentage, 'estimated_tp_time': estimated_tp_time, 'tp_probability': tp_probability})

if __name__ == '__main__':
    app.run(debug=True)
