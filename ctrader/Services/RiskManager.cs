using System;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;
using DRLTrader.Models;

namespace DRLTrader.Services
{
    /// <summary>
    /// Manages risk calculations for trading positions
    /// </summary>
    public class RiskManager
    {
        private readonly double _balancePerLot;
        private readonly double _stopLossPips;
        private readonly Symbol _symbol;
        private readonly Action<string> _logger;

        public RiskManager(double balancePerLot, double stopLossPips, Symbol symbol, Action<string> logger = null)
        {
            _balancePerLot = balancePerLot;
            _stopLossPips = stopLossPips;
            _symbol = symbol;
            _logger = logger ?? Console.WriteLine;
        }

        /// <summary>
        /// Calculate position size based on account balance using Python's exact logic
        /// </summary>
        public double CalculatePositionSize(double accountBalance)
        {
            try
            {
                // Get symbol volume step and limits
                double minVolume = _symbol.VolumeInUnitsMin;
                double maxVolume = _symbol.VolumeInUnitsMax;
                double volumeStep = _symbol.VolumeInUnitsStep;
                
                _logger($"\n==== Position Size Calculation ====");
                _logger($"Account Balance: {accountBalance:F2}");
                _logger($"Balance Per Lot: {_balancePerLot:F2}");
                _logger($"Symbol: {_symbol.Name}");
                _logger($"Min Volume: {minVolume:F3}");
                _logger($"Max Volume: {maxVolume:F3}");
                _logger($"Volume Step: {volumeStep:F3}");

                // Calculate raw position size exactly like Python
                double rawLots = accountBalance / _balancePerLot;
                _logger($"Raw Lots: {rawLots:F6}");
                
                // Round to nearest volume step
                double steps = Math.Round(rawLots / volumeStep);
                double lotSize = steps * volumeStep;
                _logger($"Rounded Steps: {steps}");
                _logger($"Initial Lot Size: {lotSize:F3}");
                
                // Ensure within min/max bounds
                lotSize = Math.Max(minVolume, Math.Min(maxVolume, lotSize));
                _logger($"Final Lot Size: {lotSize:F3}");
                _logger("============================\n");
                
                return lotSize;
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to calculate position size: {ex.Message}", ex);
            }
        }
        
        /// <summary>
        /// Calculate stop loss price for a position
        /// </summary>
        public double CalculateStopLoss(double entryPrice, TradeType tradeType)
        {
            try
            {
                _logger($"\n==== Stop Loss Calculation ====");
                _logger($"Entry Price: {entryPrice:F5}");
                _logger($"Trade Type: {tradeType}");
                _logger($"Stop Loss Pips: {_stopLossPips}");
                _logger($"Tick Size: {_symbol.TickSize}");
                _logger($"Symbol Digits: {_symbol.Digits}");

                // Convert pips to price points
                double stopLossPoints = _stopLossPips * _symbol.TickSize;
                _logger($"Stop Loss Points: {stopLossPoints:F5}");
                
                // Calculate stop loss price based on trade direction (match Python exactly)
                double stopLossPrice;
                if (tradeType == TradeType.Buy)
                {
                    stopLossPrice = entryPrice - stopLossPoints;
                    _logger($"Buy Stop Loss: Entry - Points = {entryPrice:F5} - {stopLossPoints:F5}");
                }
                else
                {
                    stopLossPrice = entryPrice + stopLossPoints;
                    _logger($"Sell Stop Loss: Entry + Points = {entryPrice:F5} + {stopLossPoints:F5}");
                }
                
                // Round to symbol digits (same as Python)
                stopLossPrice = Math.Round(stopLossPrice, _symbol.Digits);
                _logger($"Final Stop Loss Price: {stopLossPrice:F5}");
                _logger("============================\n");
                
                return stopLossPrice;
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to calculate stop loss: {ex.Message}", ex);
            }
        }
    }
}
