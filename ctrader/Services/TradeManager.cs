using System;
using System.Threading.Tasks;
using cAlgo.API;
using DRLTrader.Models;

namespace DRLTrader.Services
{
    /// <summary>
    /// Manages trade execution based on API predictions
    /// </summary>
    public class TradeManager
    {
        private readonly RiskManager _riskManager;
        private readonly Symbol _symbol;
        private readonly Robot _robot;
        private Position _currentPosition;

        public TradeManager(Robot robot, Symbol symbol, RiskManager riskManager)
        {
            _robot = robot;
            _symbol = symbol;
            _riskManager = riskManager;
        }

        /// <summary>
        /// Execute trade based on prediction
        /// </summary>
        public async Task<bool> ExecuteTradeAsync(PredictionResponse prediction)
        {
            try
            {
                // Parse the action
                if (!Enum.TryParse(prediction.Action, true, out TradingAction action))
                {
                    _robot.Print($"Invalid action received: {prediction.Action}");
                    return false;
                }

                // Update current position tracking
                _currentPosition = _robot.Positions.Find(p => p.SymbolName == _symbol.Name);

                switch (action)
                {
                    case TradingAction.Hold:
                        _robot.Print("Hold signal - no trade execution");
                        return true;

                    case TradingAction.Close:
                        return await ClosePositionAsync();

                    case TradingAction.Buy:
                    case TradingAction.Sell:
                        // Don't open new position if one exists
                        if (_currentPosition != null)
                        {
                            _robot.Print($"Trade rejected: Position already exists");
                            return true;
                        }
                        return await OpenPositionAsync(action);

                    default:
                        _robot.Print($"Unsupported action: {action}");
                        return false;
                }
            }
            catch (Exception ex)
            {
                _robot.Print($"Error executing trade: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Open a new position based on the trading action
        /// </summary>
        private async Task<bool> OpenPositionAsync(TradingAction action)
        {
            try
            {
                // Calculate position size
                double volume = _riskManager.CalculatePositionSize(_robot.Account.Balance);
                
                // Get current market price
                double entryPrice = action == TradingAction.Buy ? _symbol.Ask : _symbol.Bid;
                
                // Calculate stop loss
                TradeType tradeType = action == TradingAction.Buy ? TradeType.Buy : TradeType.Sell;
                double stopLoss = _riskManager.CalculateStopLoss(entryPrice, tradeType);
                
                // Open position
                var result = await _robot.ExecuteMarketOrderAsync(
                    tradeType,
                    _symbol,
                    volume,
                    "DRLTrader",
                    stopLoss,
                    null  // No take profit
                );
                
                if (result.Error == ErrorCode.NoError)
                {
                    _robot.Print($"Position opened: {action} {volume:F2} lots @ {entryPrice:F5} (SL: {stopLoss:F5})");
                    return true;
                }
                else
                {
                    _robot.Print($"Failed to open position: {result.Error}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _robot.Print($"Error opening position: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Close current position if it exists
        /// </summary>
        private async Task<bool> ClosePositionAsync()
        {
            try
            {
                if (_currentPosition == null)
                {
                    _robot.Print("No position to close");
                    return true;
                }

                var result = await _robot.ClosePositionAsync(_currentPosition);
                if (result.Error == ErrorCode.NoError)
                {
                    _robot.Print($"Position closed at {result.Position.ClosePrice:F5}");
                    _currentPosition = null;
                    return true;
                }
                else
                {
                    _robot.Print($"Failed to close position: {result.Error}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _robot.Print($"Error closing position: {ex.Message}");
                return false;
            }
        }
    }
}
