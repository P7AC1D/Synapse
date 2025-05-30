# Deep Reinforcement Learning Trading Bot

A sophisticated trading bot that uses Deep Reinforcement Learning (PPO-LSTM) to make autonomous trading decisions for financial markets. The system includes comprehensive training pipelines, backtesting frameworks, and live trading capabilities with advanced performance optimizations.

## 🎯 Project Overview

This project implements a complete trading system using Proximal Policy Optimization (PPO) with LSTM networks for temporal trading decision making. The bot can analyze market data, make trading decisions, and execute trades automatically while managing risk and tracking performance.

### Key Features

- **🧠 Deep Reinforcement Learning**: PPO-LSTM architecture for temporal pattern recognition
- **⚡ Performance Optimized**: 5-10x training speedup with advanced optimization techniques
- **📊 Comprehensive Backtesting**: Walk-forward validation with robust evaluation metrics
- **🔄 Live Trading**: Real-time data processing and trade execution via MetaTrader 5
- **📈 Advanced Features**: 34+ technical indicators and market regime detection
- **🛡️ Risk Management**: Built-in position sizing, stop-loss, and drawdown protection
- **📋 Extensive Logging**: Detailed trade tracking and performance analytics

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MetaTrader 5 (for live trading)
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd drl-trading-bot

# Install dependencies
cd bot/src
pip install -r requirements.txt
```

### Basic Usage

#### Training a Model

```bash
# Standard training (40-50 minutes per iteration)
python train_ppo.py --data_path ../data/XAUUSDm_15min.csv --seed 42

# Optimized training (5-10 minutes per iteration) - RECOMMENDED
python train_ppo_optimized.py --data_path ../data/XAUUSDm_15min.csv --seed 42
```

#### Live Trading

```bash
python bot.py
```

#### Backtesting

```bash
python backtest.py --model_path ../model/XAUUSDm.zip --data_path ../data/XAUUSDm_15min.csv
```

## 📁 Project Structure

```
bot/
├── data/                          # Market data files
│   ├── BTCUSDm_15min.csv         # Bitcoin/USD 15-minute data
│   ├── XAUUSDm_15min.csv         # Gold/USD 15-minute data
│   └── ...                       # Other trading pairs
├── docs/                          # Documentation
│   ├── TRAINING_OPTIMIZATION_GUIDE.md
│   ├── PERFORMANCE_IMPROVEMENT_ROADMAP.md
│   └── ...
├── model/                         # Pre-trained models
│   ├── XAUUSDm.zip               # Gold trading model
│   └── US30m.zip                 # US30 trading model
├── results/                       # Training results and checkpoints
├── src/                          # Source code
│   ├── bot.py                    # Main trading bot
│   ├── train_ppo.py              # Standard training script
│   ├── train_ppo_optimized.py    # Optimized training (5-10x faster)
│   ├── backtest.py               # Backtesting framework
│   ├── trading/                   # Trading environment and components
│   ├── utils/                     # Utilities and training functions
│   ├── callbacks/                 # Training callbacks
│   └── ...
```

## 🧠 Model Architecture

### PPO-LSTM Network

- **Policy Network**: MLP with LSTM layers for temporal decision making
- **Value Network**: Separate LSTM architecture for value estimation
- **Action Space**: 4 discrete actions (Hold, Buy, Sell, Close)
- **Observation Space**: 34+ technical indicators + position information

### Training Features

- **Walk-Forward Optimization**: Temporal validation preventing data leakage
- **Advanced Reward System**: Multi-objective optimization balancing profit and risk
- **Dynamic Position Sizing**: Adaptive lot sizing based on account balance
- **Market Regime Adaptation**: Different strategies for trending vs ranging markets

## ⚡ Performance Optimizations

### Training Speed Improvements (5-10x Speedup)

The project includes significant performance optimizations:

```bash
# Original training: 40-50 minutes per iteration
python train_ppo.py --data_path ../data/XAUUSDm_15min.csv

# Optimized training: 5-10 minutes per iteration
python train_ppo_optimized.py --data_path ../data/XAUUSDm_15min.csv
```

**Key Optimizations:**

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| **Adaptive Timesteps** | 2-4x | Reduces training steps as model matures |
| **Warm Starting** | 1.5-2x | Continues from previous best model |
| **Early Stopping** | 1.5-3x | Stops when convergence detected |
| **Progressive Training** | 1.2-1.5x | Optimizes hyperparameters by phase |
| **Environment Caching** | 1.3-1.5x | Caches preprocessing between iterations |
| **Fast Evaluation** | 10-20x | Optimized model evaluation |

### Evaluation Speed Improvements (10-20x Speedup)

Model evaluation has been dramatically optimized:

- **Batch Prediction**: Process 1000+ samples at once
- **Vectorized Calculations**: NumPy-optimized trade simulation
- **Smart Sampling**: Representative subset evaluation
- **Parallel Processing**: Multi-core model comparison

## 📊 Trading Environment

### Market Data Features

The system processes comprehensive market data including:

- **Price Data**: OHLCV (Open, High, Low, Close, Volume)
- **Technical Indicators**: 34+ indicators including RSI, Bollinger Bands, MACD, ATR
- **Market Microstructure**: Spread, slippage simulation
- **Time Features**: Session detection, volatility regimes

### Action Space

- **0 - Hold**: Maintain current position or stay flat
- **1 - Buy**: Open long position (if no position exists)
- **2 - Sell**: Open short position (if no position exists)  
- **3 - Close**: Close current position (if position exists)

### Reward System

Sophisticated multi-objective reward system optimizing:

- **Profit Maximization**: Direct PnL rewards
- **Risk Management**: Drawdown penalties
- **Trade Quality**: Reward high-probability setups
- **Market Efficiency**: Punish excessive trading

## 🔧 Configuration

### Training Parameters

Key training parameters can be configured:

```python
# Basic parameters
--data_path: Path to CSV data file
--seed: Random seed for reproducibility
--initial_balance: Starting account balance
--total_timesteps: Training steps per iteration

# Optimization parameters (optimized training)
--adaptive_timesteps: Use adaptive timestep reduction
--warm_start: Continue from previous model
--early_stopping_patience: Patience for early stopping
--progressive_training: Use progressive hyperparameters
```

### Trading Parameters

```python
--balance_per_lot: Account balance required per 0.01 lot
--point_value: Value of one price point movement
--min_lots: Minimum position size
--max_lots: Maximum position size
--contract_size: Standard contract size
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Core Metrics
- **Total Return**: Overall percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Profit Factor**: Ratio of gross profit to gross loss
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest peak-to-trough decline

### Advanced Metrics
- **Expected Value**: Average profit per trade
- **Recovery Factor**: Return divided by maximum drawdown
- **Calmar Ratio**: Annual return divided by maximum drawdown
- **Trade Distribution**: Analysis of win/loss patterns

## 🔄 Live Trading Features

### MetaTrader 5 Integration

- **Real-time Data**: Live market data fetching
- **Order Execution**: Automated trade placement
- **Position Management**: Dynamic position tracking
- **Risk Controls**: Built-in safety mechanisms

### Safety Features

- **Position Verification**: Cross-check with broker positions
- **Spread Monitoring**: Avoid trading during high spreads
- **Balance Tracking**: Real-time account monitoring
- **Error Handling**: Graceful failure recovery

## 🧪 Testing & Validation

### Comprehensive Test Suite

```bash
# Test optimized training system
python test_training_optimization.py

# Test fast evaluation system  
python test_fast_evaluation.py

# Test enhanced scoring system
python test_enhanced_scoring.py
```

### Backtesting Framework

- **Walk-Forward Validation**: Temporal train/test splits
- **Out-of-Sample Testing**: Unseen data validation
- **Monte Carlo Simulation**: Statistical robustness testing
- **Market Regime Analysis**: Performance across different market conditions

## 📚 Documentation

Comprehensive documentation is available:

- **[Training Optimization Guide](docs/TRAINING_OPTIMIZATION_GUIDE.md)**: 5-10x speedup techniques
- **[Performance Improvement Roadmap](docs/PERFORMANCE_IMPROVEMENT_ROADMAP.md)**: Enhancement strategy
- **[Fast Evaluation Guide](docs/FAST_EVALUATION_GUIDE.md)**: 10-20x evaluation speedup
- **[Phase 1 Completion Report](docs/PHASE_1_COMPLETION_REPORT.md)**: Development progress

## 🚀 Advanced Features

### Experimental Features

- **Parallel Model Training**: Train multiple candidates simultaneously
- **Hyperparameter Optimization**: Automated parameter tuning
- **Ensemble Methods**: Combine multiple models
- **Market Regime Detection**: Adaptive strategies

### Development Tools

- **Performance Profiling**: Training and inference benchmarking
- **Cache Management**: Optimization cache controls
- **Debug Mode**: Detailed logging and analysis
- **Comparison Tools**: A/B testing frameworks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading financial instruments involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.**

## 🔗 Dependencies

### Core Dependencies

```
torch>=2.6.0                 # PyTorch for neural networks
stable-baselines3>=2.5.0     # RL algorithms
sb3-contrib>=2.5.0           # Additional RL algorithms
gymnasium>=1.0.0             # RL environments
numpy>=2.0.2                 # Numerical computing
pandas>=2.2.3                # Data manipulation
ta>=0.11.0                   # Technical analysis
```

### Optional Dependencies

```
fastapi>=0.104.1             # Web API framework
uvicorn>=0.24.0              # ASGI server
onnx>=1.17.0                 # Model optimization
python-multipart>=0.0.6      # File uploads
```

## 📞 Support

For questions, issues, or contributions:

1. **Issues**: Open a GitHub issue for bugs or feature requests
2. **Documentation**: Check the `docs/` folder for detailed guides
3. **Testing**: Run the test suites to validate functionality
4. **Performance**: Use the optimization guides for best results

---

**Get started with optimized training for immediate 5-10x speedup:**

```bash
python train_ppo_optimized.py --data_path ../data/XAUUSDm_15min.csv --seed 42
```

**Experience the power of Deep Reinforcement Learning for algorithmic trading! 🚀**
