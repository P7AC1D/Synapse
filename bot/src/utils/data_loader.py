"""
Data loading utilities for trading model training.

This module provides standardized data loading and preprocessing functions
used across different training scripts. It handles CSV file loading,
data validation, timestamp conversion, and basic data cleaning.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional


def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """
    Load and prepare trading data from a CSV file.
    
    This function provides standardized data loading that is compatible
    with all training scripts in the project. It handles:
    - CSV file loading
    - Required column validation 
    - Timestamp/datetime index setting
    - NaN value handling
    - Basic data quality checks
    
    Args:
        data_path: Path to the CSV file containing trading data
        
    Returns:
        Preprocessed DataFrame with proper datetime index and validated columns
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    print(f"📊 Loading trading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        # Load CSV data
        data = pd.read_csv(data_path)
        print(f"✓ Data loaded: {len(data):,} rows, {len(data.columns)} columns")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle datetime index - try multiple common column names
        datetime_column = None
        possible_datetime_columns = ['timestamp', 'time', 'datetime', 'date']
        
        for col in possible_datetime_columns:
            if col in data.columns:
                datetime_column = col
                break
        
        if datetime_column:
            # Convert to datetime and set as index
            data[datetime_column] = pd.to_datetime(data[datetime_column])
            data = data.set_index(datetime_column)
            print(f"✓ {datetime_column.title()} index set: {data.index[0]} to {data.index[-1]}")
        else:
            # If no datetime column found, create a simple range index
            print("⚠️ No datetime column found - using sequential index")
        
        # Handle missing values
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            print(f"⚠️ Found {nan_count} NaN values - forward filling")
            data = data.fillna(method='ffill')
            
            # If still NaN values after forward fill, drop them
            remaining_nan = data.isnull().sum().sum()
            if remaining_nan > 0:
                print(f"⚠️ Dropping {remaining_nan} remaining NaN values")
                data = data.dropna()
        
        # Add spread column if not present (required for trading environment)
        if 'spread' not in data.columns:
            # Use a default spread based on price volatility
            price_volatility = data['close'].pct_change().std()
            default_spread = max(0.1, price_volatility * 100)  # At least 0.1 point spread
            data['spread'] = default_spread
            print(f"✓ Added default spread column: {default_spread:.2f}")
        
        # Basic data quality validation
        if len(data) < 100:
            raise ValueError(f"Insufficient data: need at least 100 rows, got {len(data)}")
        
        # Check for invalid price data
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] <= 0).any():
                raise ValueError(f"Invalid data: {col} contains non-positive values")
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) | 
            (data['high'] < data['open']) | 
            (data['high'] < data['close']) |
            (data['low'] > data['open']) | 
            (data['low'] > data['close'])
        ).any()
        
        if invalid_ohlc:
            print("⚠️ Warning: Some OHLC data relationships are invalid")
        
        # Log data summary
        print(f"📈 Data range: {data.index[0] if hasattr(data.index, 'min') else 'Index 0'} to {data.index[-1] if hasattr(data.index, 'max') else f'Index {len(data)-1}'}")
        print(f"📊 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"📊 Volume range: {data['volume'].min():,.0f} - {data['volume'].max():,.0f}")
        
        return data
        
    except Exception as e:
        raise ValueError(f"Error loading data from {data_path}: {e}")


def validate_data_quality(data: pd.DataFrame) -> dict:
    """
    Perform comprehensive data quality validation.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        Dictionary containing validation results and statistics
    """
    results = {
        'total_rows': len(data),
        'missing_values': data.isnull().sum().to_dict(),
        'data_types': data.dtypes.to_dict(),
        'warnings': [],
        'errors': []
    }
    
    # Check for required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check data ranges
    if 'close' in data.columns:
        price_stats = {
            'min_price': data['close'].min(),
            'max_price': data['close'].max(),
            'price_volatility': data['close'].pct_change().std()
        }
        results['price_statistics'] = price_stats
        
        if price_stats['min_price'] <= 0:
            results['errors'].append("Close prices contain non-positive values")
    
    # Check for gaps in data
    if hasattr(data.index, 'to_series'):
        time_diffs = data.index.to_series().diff()
        if len(time_diffs.unique()) > 10:  # Too many different intervals
            results['warnings'].append("Irregular time intervals detected")
    
    return results


def load_data_with_validation(data_path: str, strict_validation: bool = True) -> pd.DataFrame:
    """
    Load data with comprehensive validation and error reporting.
    
    Args:
        data_path: Path to data file
        strict_validation: If True, raise errors on validation failures
        
    Returns:
        Loaded and validated DataFrame
    """
    data = load_and_prepare_data(data_path)
    validation_results = validate_data_quality(data)
    
    # Report validation results
    if validation_results['warnings']:
        for warning in validation_results['warnings']:
            print(f"⚠️ Warning: {warning}")
    
    if validation_results['errors']:
        error_msg = "Data validation errors:\n" + "\n".join(validation_results['errors'])
        if strict_validation:
            raise ValueError(error_msg)
        else:
            print(f"❌ {error_msg}")
    
    return data
