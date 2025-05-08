using System;
using System.Threading.Tasks;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;
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

                // Get ALL positions for the symbol to ensure we're finding everything
                var positions = _robot.Positions.FindAll(_symbol.Name);
                int positionCount = positions.Count;
                
                if (positionCount > 0)
                {
                    _robot.Print($"Found {positionCount} existing positions for {_symbol.Name}");
                    foreach (var pos in positions)
                    {
                        _robot.Print($"Position: {pos.Id}, {pos.TradeType}, {pos.Volume} lots, PnL: {pos.NetProfit}");
                    }
                    
                    // Update current position to the first one found
                    _currentPosition = positions.FirstOrDefault();
                }
                else
                {
                    _robot.Print($"No existing positions found for {_symbol.Name}");
                    _currentPosition = null;
                }

                switch (action)
                {
                    case TradingAction.Hold:
                        _robot.Print("Hold signal - no trade execution");
                        return true;

                    case TradingAction.Close:
                        if (positionCount > 0)
                        {
                            // Close all positions for this symbol
                            return await CloseAllPositionsAsync(positions);
                        }
                        else
                        {
                            _robot.Print("No positions to close");
                            return true;
                        }

                    case TradingAction.Buy:
                        // If we have a sell position, close it first
                        if (positionCount > 0 && _currentPosition.TradeType == TradeType.Sell)
                        {
                            _robot.Print("Closing existing Sell positions before opening Buy position");
                            await CloseAllPositionsAsync(positions);
                            return await OpenPositionAsync(action);
                        }
                        // If we already have a buy position, don't open another one
                        else if (positionCount > 0 && _currentPosition.TradeType == TradeType.Buy)
                        {
                            _robot.Print($"Buy signal received but already have {positionCount} Buy positions - holding");
                            return true;
                        }
                        // No positions, open a new one
                        else
                        {
                            return await OpenPositionAsync(action);
                        }
                        
                    case TradingAction.Sell:
                        // If we have a buy position, close it first
                        if (positionCount > 0 && _currentPosition.TradeType == TradeType.Buy)
                        {
                            _robot.Print("Closing existing Buy positions before opening Sell position");
                            await CloseAllPositionsAsync(positions);
                            return await OpenPositionAsync(action);
                        }
                        // If we already have a sell position, don't open another one
                        else if (positionCount > 0 && _currentPosition.TradeType == TradeType.Sell)
                        {
                            _robot.Print($"Sell signal received but already have {positionCount} Sell positions - holding");
                            return true;
                        }
                        // No positions, open a new one
                        else
                        {
                            return await OpenPositionAsync(action);
                        }

                    default:
                        _robot.Print($"Unsupported action: {action}");
                        return false;
                }
            }
            catch (Exception ex)
            {
                _robot.Print($"Error executing trade: {ex.Message}");
                _robot.Print($"Stack trace: {ex.StackTrace}");
                return false;
            }
        }

        /// <summary>
        /// Open a new position based on the trading action
        /// </summary>
        private Task<bool> OpenPositionAsync(TradingAction action)
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
                var result = _robot.ExecuteMarketOrder(
                    tradeType,
                    _symbol.Name,
                    volume,
                    "DRLTrader",
                    stopLoss,
                    null  // No take profit
                );
                
                if (result.IsSuccessful)
                {
                    _robot.Print($"Position opened: {action} {volume:F2} lots @ {entryPrice:F5} (SL: {stopLoss:F5})");
                    return Task.FromResult(true);
                }
                else
                {
                    _robot.Print($"Failed to open position: {result.Error}");
                    return Task.FromResult(false);
                }
            }
            catch (Exception ex)
            {
                _robot.Print($"Error opening position: {ex.Message}");
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Close current position if it exists
        /// </summary>
        private Task<bool> ClosePositionAsync()
        {
            try
            {
                if (_currentPosition == null)
                {
                    _robot.Print("No position to close");
                    return Task.FromResult(true);
                }

                var result = _robot.ClosePosition(_currentPosition);
                if (result.IsSuccessful)
                {
                    _robot.Print($"Position closed");
                    _currentPosition = null;
                    return Task.FromResult(true);
                }
                else
                {
                    _robot.Print($"Failed to close position: {result.Error}");
                    return Task.FromResult(false);
                }
            }
            catch (Exception ex)
            {
                _robot.Print($"Error closing position: {ex.Message}");
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Close all positions for the given symbol
        /// </summary>
        private async Task<bool> CloseAllPositionsAsync(Position[] positions)
        {
            try
            {
                foreach (var position in positions)
                {
                    var result = _robot.ClosePosition(position);
                    if (!result.IsSuccessful)
                    {
                        _robot.Print($"Failed to close position {position.Id}: {result.Error}");
                        return false;
                    }
                }
                _robot.Print("All positions closed");
                _currentPosition = null;
                return true;
            }
            catch (Exception ex)
            {
                _robot.Print($"Error closing all positions: {ex.Message}");
                return false;
            }
        }
    }
}
