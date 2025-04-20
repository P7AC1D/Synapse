import requests
import json
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(bars=500, start_price=1800.0, volatility=0.5):
    """Generate realistic-looking OHLCV data for testing."""
    base_time = int(datetime.now().timestamp())
    price = start_price
    data = {
        "timestamp": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": [],
        "symbol": "XAUUSDm"
    }
    
    for i in range(bars):
        # Generate realistic price movement
        price_change = (np.random.random() - 0.5) * volatility
        price += price_change
        
        # Generate OHLCV data
        open_price = price
        high_price = price + abs(np.random.random() * volatility)
        low_price = price - abs(np.random.random() * volatility)
        close_price = price + (np.random.random() - 0.5) * volatility
        volume = 1000 + np.random.random() * 1000
        
        # Add to data
        data["timestamp"].append(base_time + i * 300)  # 5-minute bars
        data["open"].append(round(open_price, 2))
        data["high"].append(round(max(high_price, open_price, close_price), 2))
        data["low"].append(round(min(low_price, open_price, close_price), 2))
        data["close"].append(round(close_price, 2))
        data["volume"].append(round(volume, 0))
    
    return data

def test_health():
    """Test the health check endpoint."""
    try:
        response = requests.get("http://localhost:8000/health")
        logger.info("\nHealth Check:")
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {response.json()}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise

def test_predict():
    """Test the prediction endpoint with different scenarios."""
    try:
        # Test 1: Normal prediction with sufficient data
        logger.info("\nTest 1: Normal prediction (200 bars)")
        data = generate_sample_data(bars=200)
        response = requests.post("http://localhost:8000/predict", json=data)
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        assert "action" in response.json()
        assert "confidence" in response.json()
        assert "timestamp" in response.json()
        assert "description" in response.json()
        
        # Test 2: Insufficient data (should fail with 400)
        logger.info("\nTest 2: Insufficient data (50 bars)")
        min_data = generate_sample_data(bars=50)
        response = requests.post("http://localhost:8000/predict", json=min_data)
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 400
        assert "Insufficient data" in response.json()["detail"], f"Unexpected error message: {response.json()['detail']}"
        assert "minimum required is 100" in response.json()["detail"]
        
        # Test 3: Invalid symbol (should return 404)
        logger.info("\nTest 3: Invalid symbol")
        invalid_data = generate_sample_data(bars=200)
        invalid_data["symbol"] = "INVALID"
        response = requests.post("http://localhost:8000/predict", json=invalid_data)
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 404
        assert response.json()["detail"] == f"Model not found for symbol {invalid_data['symbol']}"
        
        logger.info("\nAll prediction tests passed successfully!")
        
    except AssertionError as e:
        logger.error(f"Prediction test failed: Assertion Error - {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Prediction test failed: {str(e)}")
        raise

if __name__ == "__main__":
    import numpy as np  # For random data generation
    
    logger.info("Starting API tests...")
    test_health()
    test_predict()
    logger.info("All tests completed successfully!")
