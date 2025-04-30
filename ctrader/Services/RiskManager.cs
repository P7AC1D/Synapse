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
        
        public RiskManager(double balancePerLot, double stopLossPips, Symbol symbol)
        {
            _balancePerLot = balancePerLot;
            _stopLossPips = stopLossPips;
            _symbol = symbol;
        }
        
        /// <summary>
        /// Calculate position size based on account balance
        /// </summary>
        public double CalculatePositionSize(double accountBalance)
        {
            try
            {
                // Get symbol volume step and limits
                double minVolume = _symbol.VolumeInUnitsMin;
                double maxVolume = _symbol.VolumeInUnitsMax;
                double volumeStep = _symbol.VolumeInUnitsStep;
                
                // Calculate raw position size (same logic as Python implementation)
                double rawLotSize = (accountBalance / _balancePerLot) * volumeStep;
                
                // Round to nearest volume step
                int steps = (int)Math.Round(rawLotSize / volumeStep);
                double lotSize = steps * volumeStep;
                
                // Ensure within min/max bounds
                lotSize = Math.Max(minVolume, Math.Min(maxVolume, lotSize));
                
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
                // Convert pips to price points based on symbol digits
                double stopLossPoints = _stopLossPips * _symbol.TickSize;
                
                // Calculate stop loss price based on trade direction
                double stopLossPrice = tradeType == TradeType.Buy 
                    ? entryPrice - stopLossPoints 
                    : entryPrice + stopLossPoints;
                
                // Round to symbol digits
                stopLossPrice = Math.Round(stopLossPrice, _symbol.Digits);
                
                return stopLossPrice;
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to calculate stop loss: {ex.Message}", ex);
            }
        }
    }
}
