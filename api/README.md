# Trading Prediction API

A FastAPI-based web service that provides trading predictions using trained DRL models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
cd src
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /health
Health check endpoint to verify the API is running.

Response:
```json
{
    "status": "healthy"
}
```

### POST /predict
Get trading prediction for a given market data series.

Request Body:
```json
{
    "timestamp": [1634567890, 1634567891, ...],
    "open": [1800.5, 1801.2, ...],
    "high": [1802.1, 1803.4, ...],
    "low": [1799.8, 1800.9, ...],
    "close": [1801.3, 1802.7, ...],
    "volume": [1234, 5678, ...],
    "symbol": "XAUUSDm"
}
```

Response:
```json
{
    "action": "buy",
    "confidence": 0.95,
    "timestamp": 1634567891
}
```

## Example Usage

```python
import requests
import json

url = "http://localhost:8000/predict"
data = {
    "timestamp": [1634567890, 1634567891],
    "open": [1800.5, 1801.2],
    "high": [1802.1, 1803.4],
    "low": [1799.8, 1800.9],
    "close": [1801.3, 1802.7],
    "volume": [1234, 5678],
    "symbol": "XAUUSDm"
}

response = requests.post(url, json=data)
prediction = response.json()
print(prediction)
