from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the bot source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../bot/src"))

from trading.features import FeatureProcessor
from trading.actions import Action
from trade_model import TradeModel
from trading.environment import TradingEnv

app = FastAPI(
    title="Trading Prediction API",
    description="""
    This API provides trading predictions using a trained Deep Reinforcement Learning model.
    It accepts market data and returns recommended trading actions.
    """,
    version="1.0.0"
)

class MarketData(BaseModel):
    timestamp: list[int] = Field(..., description="List of Unix timestamps for each bar")
    open: list[float] = Field(..., description="List of opening prices")
    high: list[float] = Field(..., description="List of high prices")
    low: list[float] = Field(..., description="List of low prices")
    close: list[float] = Field(..., description="List of closing prices")
    volume: list[float] = Field(..., description="List of volume values")
    symbol: str = Field(..., description="Trading symbol (e.g., 'XAUUSDm')")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": [1634567890, 1634567950],
                "open": [1800.5, 1801.2],
                "high": [1802.1, 1803.4],
                "low": [1799.8, 1800.9],
                "close": [1801.3, 1802.7],
                "volume": [1234.0, 5678.0],
                "symbol": "XAUUSDm"
            }
        }

class PredictionResponse(BaseModel):
    action: str = Field(..., description="Recommended trading action (buy, sell, hold, or close)")
    confidence: float = Field(..., description="Confidence score for the prediction")
    timestamp: int = Field(..., description="Timestamp of the prediction")
    description: str = Field(..., description="Detailed description of the prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "action": "hold",
                "confidence": 0.95,
                "timestamp": 1634567950,
                "description": "Model predicts hold"
            }
        }

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the API is healthy and ready to accept requests."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse, tags=["Trading"])
async def predict(data: MarketData):
    """
    Get a trading prediction based on market data.
    
    This endpoint accepts OHLCV market data and returns a recommended trading action.
    The model requires at least 100 bars of data for accurate predictions.
    """
    try:
        # Validate input data length
        min_bars = 100  # Minimum required bars for prediction
        if len(data.timestamp) < min_bars:
            error_msg = f"Insufficient data: provided {len(data.timestamp)} bars, minimum required is {min_bars}"
            logger.error(error_msg)
            return JSONResponse(
                status_code=400,
                content={"detail": error_msg}
            )

        # Convert input data to DataFrame and add spread column
        df = pd.DataFrame({
            'timestamp': data.timestamp,
            'open': data.open,
            'high': data.high,
            'low': data.low,
            'close': data.close,
            'volume': data.volume,
            'spread': [0.00001] * len(data.timestamp)  # Adding minimal spread as it's required by the model
        })
        
        # Set timestamp as index (required by feature processor)
        df.set_index('timestamp', inplace=True)
        
        # Validate and locate model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../bot/model/{data.symbol}.zip"))
        if not os.path.exists(model_path):
            error_msg = f"Model not found for symbol {data.symbol}"
            logger.error(f"{error_msg}: {model_path}")
            # Raise an exception that will be caught by the outer try-except block
            raise FileNotFoundError(error_msg)
        
        # Initialize model and processors
        try:
            trade_model = TradeModel(model_path)
        except Exception as model_error:
            logger.error(f"Failed to create model instance: {str(model_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(model_error)}")
        if not trade_model.load_model():
            raise HTTPException(status_code=500, detail="Failed to load model")
        
        # Create environment for observation (use the last window_size bars)
        window_data = df.iloc[-trade_model.window_size:].copy() if len(df) > trade_model.window_size else df.copy()
        
        # Initialize LSTM states if needed
        if trade_model.lstm_states is None and len(df) >= trade_model.initial_warmup:
            logger.info("Initializing LSTM states with warmup data")
            warmup_data = df.iloc[:trade_model.initial_warmup].copy()
            trade_model.preload_states(warmup_data)
        
        # Create environment for observation
        env = TradingEnv(
            data=window_data,
            initial_balance=trade_model.initial_balance,
            balance_per_lot=trade_model.balance_per_lot,
            random_start=False,
            point_value=trade_model.point_value,
            min_lots=trade_model.min_lots,
            max_lots=trade_model.max_lots,
            contract_size=trade_model.contract_size
        )
        obs, _ = env.reset()
        
        # Make prediction using direct model.predict like bot and backtest
        action, new_lstm_states = trade_model.model.predict(
            obs,
            state=trade_model.lstm_states,
            deterministic=True
        )
        trade_model.lstm_states = new_lstm_states
        
        # Convert action to discrete value
        try:
            if isinstance(action, np.ndarray):
                action_value = int(action.item())
            else:
                action_value = int(action)
            discrete_action = action_value % 4
        except (ValueError, TypeError):
            discrete_action = 0
        
        # Get position type (always 0 for API since we don't track positions)
        position_type = 0
        
        # Generate description
        description = trade_model._generate_prediction_description(discrete_action, position_type)
        
        # Get action details
        action_obj = Action(discrete_action)
        action_name = action_obj.name.lower()
        
        logger.info(f"Prediction made: {action_name} ({description})")
        
        # Return prediction
        return PredictionResponse(
            action=action_name,
            confidence=0.95,  # TODO: Implement confidence calculation
            timestamp=data.timestamp[-1],
            description=description
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error making prediction: {error_msg}")

        if "Model not found" in error_msg:
            raise HTTPException(status_code=404, detail=error_msg)
        elif "Lookback window" in error_msg:
            raise HTTPException(status_code=400, detail=error_msg)
        else:
            raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
