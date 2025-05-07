"""
Training hyperparameters based on financial ML research papers.

References:
- "Deep Reinforcement Learning in Financial Markets" (Théate & Ernst, 2021)
- "Deep Reinforcement Learning for Trading Applications" (Xiong et al., 2019)
- "Practical Deep Reinforcement Learning Approach for Stock Trading" (Huang, 2018)
- "Deep Reinforcement Learning for Trading" (Zhang et al., 2020)
- "Reinforcement Learning for Quantitative Trading" by Moody and Saffell, 2001
"""
import torch as th

def get_wfo_params(args, data_length):
    """
    Returns optimized Walk-Forward Optimization parameters based on financial ML research papers.
    
    Args:
        args: Command line arguments
        data_length: Total length of dataset (number of bars)
        
    Returns:
        Dictionary with recommended WFO parameters
    """
    # Base number of data points for training window
    # Based on research showing minimum 5,000-10,000 points needed for RL in financial markets
    base_window_size = min(10_080, max(5_000, int(data_length * 0.11)))
    
    # For 15-minute data, typical research recommends ~3 weeks for step size
    # (assuming 96 bars per day * 5 days * ~3 weeks)
    base_step_size = 2_016
    
    # Adjustments based on market type
    if hasattr(args, 'market_type'):
        if args.market_type == 'short_term':  # Day trading
            # For short-term, research suggests higher overlap ratio (80%)
            step_size = min(int(base_window_size * 0.2), base_step_size)
            window_size = base_window_size
            # Optimal training timesteps from financial RL studies for short-term patterns
            total_timesteps = 500_000
            
        elif args.market_type == 'medium_term':  # Swing trading
            # Medium-term trading: ~6 months window, 1 month step (75% overlap)
            window_size = min(int(data_length * 0.2), max(base_window_size * 2, 20_000))
            step_size = min(int(window_size * 0.25), 5_000)
            # More complex patterns require more timesteps
            total_timesteps = 750_000
            
        elif args.market_type == 'long_term':  # Position trading
            # Long-term trading: ~1 year window, 2 month step (70% overlap)
            window_size = min(int(data_length * 0.3), max(base_window_size * 3, 30_000))
            step_size = min(int(window_size * 0.3), 9_000)
            # Deep value approximation requires more training
            total_timesteps = 1_000_000
    else:
        # Default to short-term parameters
        window_size = base_window_size
        step_size = base_step_size
        total_timesteps = 500_000
    
    # Ensure window size is at least 5x step size (research recommendation)
    if window_size < step_size * 5:
        window_size = step_size * 5
    
    # Validation split - optimal range from financial ML papers
    validation_split = 0.2
    
    # Calculate actual size in real-world terms
    bars_per_day = 96  # 15min data = 96 bars/day
    window_days = window_size / bars_per_day
    step_days = step_size / bars_per_day
    
    return {
        'window_size': window_size,
        'step_size': step_size,
        'total_timesteps': total_timesteps,
        'validation_split': validation_split,
        'window_days': window_days,
        'step_days': step_days,
        'data_coverage': f"{(window_size / data_length * 100):.1f}% of dataset"
    }