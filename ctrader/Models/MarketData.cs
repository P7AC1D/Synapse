using System;
using System.Collections.Generic;

namespace DRLTrader.Models
{
    /// <summary>
    /// Market data structure matching the API request format
    /// </summary>
    public class MarketData
    {
        public List<long> Timestamp { get; set; }
        public List<double> Open { get; set; }
        public List<double> High { get; set; }
        public List<double> Low { get; set; }
        public List<double> Close { get; set; }
        public List<double> Volume { get; set; }
        public string Symbol { get; set; }
        public int PositionDirection { get; set; }
        public double PositionPnl { get; set; }

        public MarketData()
        {
            Timestamp = new List<long>();
            Open = new List<double>();
            High = new List<double>();
            Low = new List<double>();
            Close = new List<double>();
            Volume = new List<double>();
            PositionDirection = 0;
            PositionPnl = 0.0;
        }
    }

    /// <summary>
    /// API prediction response structure
    /// </summary>
    public class PredictionResponse
    {
        public string Action { get; set; }
        public double Confidence { get; set; }
        public long Timestamp { get; set; }
        public string Description { get; set; }
    }

    /// <summary>
    /// Enum representing possible trading actions
    /// </summary>
    public enum TradingAction
    {
        Hold = 0,
        Buy = 1,
        Sell = 2,
        Close = 3
    }
}
