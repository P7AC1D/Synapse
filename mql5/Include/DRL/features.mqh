// Feature processing for DRL model
// Generated on: 2025-04-18 05:05:25

#include <Trade/Trade.mqh>
#include <Arrays/ArrayDouble.mqh>
#include <Math/Stat/Math.mqh>

#property copyright "Copyright 2024, DRL Trading Bot"
#property link      "https://github.com/your-repo"
#property version   "1.00"

#ifndef _DRL_FEATURES_H_
#define _DRL_FEATURES_H_

// Indicator Parameters
#define RSI_PERIOD 14
#define ATR_PERIOD 14
#define BOLL_PERIOD 20

// Time constants
#define MINUTES_IN_DAY 1440

class CFeatureProcessor {
private:
    int m_atr_handle;
    int m_rsi_handle;
    int m_bb_handle;
    int m_adx_handle;

public:
    void Init(const string symbol, const ENUM_TIMEFRAMES timeframe) {
        m_atr_handle = iATR(symbol, timeframe, ATR_PERIOD);
        m_rsi_handle = iRSI(symbol, timeframe, RSI_PERIOD, PRICE_CLOSE);
        m_bb_handle = iBands(symbol, timeframe, BOLL_PERIOD, 0, 2, PRICE_CLOSE);
        m_adx_handle = iADX(symbol, timeframe, ATR_PERIOD);
    }

    void ProcessFeatures(double& features[]) {
        double close[];
        double open[];
        double high[];
        double low[];
        long volume[];
        ArraySetAsSeries(close, true);
        ArraySetAsSeries(open, true);
        ArraySetAsSeries(high, true);
        ArraySetAsSeries(low, true);
        ArraySetAsSeries(volume, true);


        // Get price data
        CopyClose(_Symbol, _Period, 0, 2, close);
        CopyOpen(_Symbol, _Period, 0, 1, open);
        CopyHigh(_Symbol, _Period, 0, 1, high);
        CopyLow(_Symbol, _Period, 0, 1, low);
        CopyTickVolume(_Symbol, _Period, 0, 2, volume);

        // Calculate returns
        double returns = (close[0] - close[1]) / close[1];
        returns = MathMax(MathMin(returns, 0.1), -0.1);

        // Get indicators
        double atr[], rsi[], bb_upper[], bb_lower[], adx[];
        ArraySetAsSeries(atr, true);
        ArraySetAsSeries(rsi, true);
        ArraySetAsSeries(bb_upper, true);
        ArraySetAsSeries(bb_lower, true);
        ArraySetAsSeries(adx, true);

        CopyBuffer(m_atr_handle, 0, 0, 1, atr);
        CopyBuffer(m_rsi_handle, 0, 0, 1, rsi);
        CopyBuffer(m_bb_handle, 1, 0, 1, bb_upper);
        CopyBuffer(m_bb_handle, 2, 0, 1, bb_lower);
        CopyBuffer(m_adx_handle, 0, 0, 1, adx);

        // Normalize RSI to [-1, 1]
        double norm_rsi = rsi[0] / 50.0 - 1.0;

        // Normalize ATR using historical window
        double atr_window[], close_history[];
        ArraySetAsSeries(atr_window, true);
        ArraySetAsSeries(close_history, true);

        // Copy historical data for ATR normalization
        int history_window = 100;  // Match Python's historical window
        CopyBuffer(m_atr_handle, 0, 0, history_window, atr_window);
        CopyClose(_Symbol, _Period, 0, history_window, close_history);

        // Calculate ATR/close ratios
        double atr_close_ratios[];
        ArrayResize(atr_close_ratios, history_window);
        double min_ratio = DBL_MAX;
        double max_ratio = -DBL_MAX;

        for(int i=0; i<history_window; i++) {
            if(close_history[i] > 0) {
                double ratio = atr_window[i] / close_history[i];
                atr_close_ratios[i] = ratio;
                min_ratio = MathMin(min_ratio, ratio);
                max_ratio = MathMax(max_ratio, ratio);
            }
        }

        // Current ATR/close ratio
        double current_ratio = atr[0] / close[0];

        // Normalize using historical min/max
        double norm_atr = 2.0 * (current_ratio - min_ratio) / (max_ratio - min_ratio + 1e-8) - 1.0;

        // Calculate volatility breakout
        double band_range = bb_upper[0] - bb_lower[0];
        double position = close[0] - bb_lower[0];
        double volatility_breakout = position / (band_range + 1e-8);
        volatility_breakout = MathMax(MathMin(volatility_breakout, 1.0), 0.0);

        // Calculate trend strength
        double trend_strength = MathMax(MathMin(adx[0]/25.0 - 1.0, 1.0), -1.0);

        // Calculate candle pattern
        double body = close[0] - open[0];
        double upper_wick = high[0] - MathMax(close[0], open[0]);
        double lower_wick = MathMin(close[0], open[0]) - low[0];
        double range = high[0] - low[0] + 1e-8;
        double candle_pattern = (body/range + 
                               (upper_wick - lower_wick)/(upper_wick + lower_wick + 1e-8)) / 2.0;
        candle_pattern = MathMax(MathMin(candle_pattern, 1.0), -1.0);

        // Calculate time features
        MqlDateTime time;
        TimeToStruct(TimeCurrent(), time);
        int minutes = time.hour * 60 + time.min;
        double sin_time = MathSin(2.0 * M_PI * minutes / MINUTES_IN_DAY);
        double cos_time = MathCos(2.0 * M_PI * minutes / MINUTES_IN_DAY);

        // Calculate volume change
        double volume_change = 0.0;
        if(volume[1] > 0) {
            volume_change = ((double)volume[0] - (double)volume[1]) / (double)volume[1];
            volume_change = MathMax(MathMin(volume_change, 1.0), -1.0);
        }

        // Set features array
        ArrayResize(features, 9);  // Base features (position features added separately)
        features[0] = returns;
        features[1] = norm_rsi;
        features[2] = norm_atr;
        features[3] = volume_change;
        features[4] = volatility_breakout;
        features[5] = trend_strength;
        features[6] = candle_pattern;
        features[7] = sin_time;
        features[8] = cos_time;
    }

    void Deinit() {
        IndicatorRelease(m_atr_handle);
        IndicatorRelease(m_rsi_handle);
        IndicatorRelease(m_bb_handle);
        IndicatorRelease(m_adx_handle);
    }
};

#endif  // _DRL_FEATURES_H_
