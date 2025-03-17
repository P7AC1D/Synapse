import pandas as pd
from trade_model import TradeModel

# Load test data
data = pd.read_csv('../data/BTCUSDm_60min.csv')
data.set_index('time', inplace=True)
data.drop(columns=['EMA_medium', 'MACD', 'Stoch', 'BB_upper', 'BB_middle', 'BB_lower'], inplace=True)

# Print data information
start_datetime = data.index[0]
end_datetime = data.index[-1]
print(f"Data collected from {start_datetime} to {end_datetime}")
print(data.tail())
print('\n')

# Remove rows with NaN values
data.dropna(inplace=True)

# Create trade model with our best hyperparameters
model = TradeModel(
    model_path='../results/65478/best_balance_model.zip',
    bar_count=50,
    normalization_window=100
)

# Run a backtest
backtest_data = data.iloc[-10000:]  # Use last 5000 bars for backtest
results = model.backtest(backtest_data)
print(f"\nBacktest Results:")
print(f"Final Balance: ${results['final_balance']:.2f}")
print(f"Return: {results['return_pct']:.2f}%")
print(f"Total Trades: {results['total_trades']}")
print(f"Win Rate: {results['win_rate']:.2f}%")
print(f"Profit Factor: {results['profit_factor']:.2f}")