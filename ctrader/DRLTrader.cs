using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Indicators;
using DRLTrader.Models;
using DRLTrader.Services;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class DRLTrader : Robot
    {
        [Parameter("API URL", DefaultValue = "http://localhost:8000", Group = "API")]
        public string ApiUrl { get; set; }

        [Parameter("Balance Per Lot", DefaultValue = 1000.0, Group = "Risk")]
        public double BalancePerLot { get; set; }

        [Parameter("Stop Loss Pips", DefaultValue = 2500.0, Group = "Risk")]
        public double StopLossPips { get; set; }

        [Parameter("Minimum Bars", DefaultValue = 100, Group = "Data")]
        public int MinimumBars { get; set; }

        private ApiClient _apiClient;
        private RiskManager _riskManager;
        private TradeManager _tradeManager;
        private readonly Queue<Bar> _bars;
        private bool _isInitialized;

        public DRLTrader()
        {
            _bars = new Queue<Bar>();
        }

        protected override void OnStart()
        {
            try
            {
                Print("Starting DRLTrader initialization...");
                
                // Initialize services
                Print($"Initializing API client with URL: {ApiUrl}");
                _apiClient = new ApiClient(ApiUrl);
                
                Print($"Initializing Risk Manager with BalancePerLot: {BalancePerLot}, StopLossPips: {StopLossPips}");
                _riskManager = new RiskManager(BalancePerLot, StopLossPips, Symbol);
                
                Print("Initializing Trade Manager");
                _tradeManager = new TradeManager(this, Symbol, _riskManager);

                // Check API health
                Print("Checking API health...");
                CheckApiHealth().Wait();

                // Load historical data
                Print($"Loading initial {MinimumBars} historical bars...");
                LoadHistoricalData();

                _isInitialized = true;
                Print("DRLTrader initialized successfully");
            }
            catch (Exception ex)
            {
                Print($"Initialization failed: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
            }
        }

        protected override void OnBar()
        {
            if (!_isInitialized)
            {
                Print("OnBar skipped - Robot not initialized");
                return;
            }

            try
            {
                Print($"Processing new bar at {Time}");
                
                // Add new bar to queue
                _bars.Enqueue(Bars.Last());
                if (_bars.Count > MinimumBars)
                    _bars.Dequeue();

                // Only proceed if we have enough data
                if (_bars.Count < MinimumBars)
                {
                    Print($"Waiting for more data: {_bars.Count}/{MinimumBars} bars");
                    return;
                }

                Print("Getting current position info...");
                // Get current position info
                var position = Positions.Find(Symbol.Name);
                int positionDirection = position != null 
                    ? (position.TradeType == TradeType.Buy ? 1 : -1) 
                    : 0;
                double positionPnl = position?.NetProfit ?? 0.0;
                Print($"Current position: Direction={positionDirection}, PnL={positionPnl}");

                // Create market data
                Print("Preparing market data...");
                var marketData = new MarketData
                {
                    Symbol = Symbol.Name,
                    PositionDirection = positionDirection,
                    PositionPnl = positionPnl
                };

                // Add historical data
                foreach (var bar in _bars)
                {
                    marketData.Timestamp.Add(new DateTimeOffset(bar.OpenTime).ToUnixTimeSeconds());
                    marketData.Open.Add(bar.Open);
                    marketData.High.Add(bar.High);
                    marketData.Low.Add(bar.Low);
                    marketData.Close.Add(bar.Close);
                    marketData.Volume.Add(bar.TickVolume);
                }
                Print($"Market data prepared with {_bars.Count} bars");

                // Get prediction
                Print("Getting prediction and executing trade...");
                GetPredictionAndExecute(marketData).Wait();
            }
            catch (Exception ex)
            {
                Print($"Error in OnBar: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
            }
        }

        protected override void OnStop()
        {
            _apiClient?.Dispose();
            Print("DRLTrader stopped");
        }

        private async Task CheckApiHealth()
        {
            if (!await _apiClient.CheckHealthAsync())
                throw new Exception("API health check failed");
            Print("API health check passed");
        }

        private void LoadHistoricalData()
        {
            // Load initial historical bars
            var historicalBars = Bars.Take(MinimumBars).ToList();
            foreach (var bar in historicalBars)
            {
                _bars.Enqueue(bar);
            }
            Print($"Loaded {_bars.Count} historical bars");
        }

        private async Task GetPredictionAndExecute(MarketData data)
        {
            try
            {
                Print("Requesting prediction from API...");
                // Get prediction from API
                var prediction = await _apiClient.GetPredictionAsync(data);
                Print($"Received prediction: {prediction.Action} ({prediction.Description})");

                // Execute trade based on prediction
                Print("Executing trade based on prediction...");
                bool success = await _tradeManager.ExecuteTradeAsync(prediction);
                if (!success)
                    Print("Trade execution failed");
                else
                    Print("Trade execution successful");
            }
            catch (Exception ex)
            {
                Print($"Error getting prediction or executing trade: {ex.Message}");
                Print($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}
