This project implements a Deep Reinforcement Learning (DRL) trading system using the following key components:

Core Architecture:
- RecurrentPPO model architecture for temporal pattern recognition in forex markets
- Walk-Forward Optimization (WFO) for robust model training and validation
- Multi-timeframe analysis focused on 15-minute intervals

Key Components:
1. Training Pipeline (train_ppo.py):
   - Implements RecurrentPPO training with WFO
   - Handles multi-currency pair training data
   - Includes comprehensive evaluation metrics

2. Backtesting System (backtest.py):
   - Historical data validation
   - Performance metrics calculation
   - Risk management verification

3. Live Trading Integration (bot.py):
   - MT5 API integration for execution
   - Real-time market data processing
   - Risk management enforcement

Operational Requirements:
- Must maintain strict risk management parameters
- Requires monitoring of model drift and performance
- Needs robustness against market regime changes
- Must handle data gaps and market closures gracefully

Development Guidelines:
- All changes must preserve model stability
- Risk management logic cannot be bypassed
- Performance impact must be validated via backtesting
- Changes must be compatible with both MT5 and cTrader platforms

Always validate changes against these criteria before implementation.
