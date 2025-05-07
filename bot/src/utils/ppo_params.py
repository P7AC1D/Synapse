"""PPO hyperparameters for trading model."""
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CallbackList

def get_ppo_params(args):
    """
    Returns optimized PPO hyperparameters based on research papers for financial trading.
    
    References:
    - "Proximal Policy Optimization Algorithms" by Schulman et al.
    - "Deep Reinforcement Learning for Trading" by Zhang et al.
    - "Optimizing Trading Strategies Using Deep Reinforcement Learning" by Fischer et al.
    - "Applications of Deep Reinforcement Learning in Financial Markets" by Mosavi et al.
    - "Deep Reinforcement Learning for Time Series: Playing Idealized Trading Games" by Dempster et al.
    """
    # Network architecture
    policy_kwargs = {
        "net_arch": {
            "pi": [128, 128, 64],  # Policy network: 3 layers
            "vf": [128, 128, 64],  # Value function: 3 layers
        },
        "activation_fn": th.nn.ReLU
    }
    
    # Core PPO parameters - baseline configuration
    ppo_params = {
        "policy": "MlpPolicy",  # Use string instead of class reference for compatibility
        "policy_kwargs": policy_kwargs,
        "learning_rate": args.learning_rate,  # Scheduled learning rate decay handled in training loop
        "n_steps": 1024,            # Update after collecting 1024 steps
        "batch_size": 256,          # Mini-batch size for gradient updates
        "n_epochs": 10,             # Number of policy update epochs
        "gamma": 0.995,             # Discount factor (higher for trading to give weight to future rewards)
        "gae_lambda": 0.95,         # GAE Lambda parameter
        "clip_range": 0.2,          # Clip parameter for PPO
        "clip_range_vf": 0.2,       # Value function clipping
        "normalize_advantage": True, # Normalize advantage
        "ent_coef": 0.01,           # Entropy coefficient for exploration (lower for trading)
        "vf_coef": 0.5,             # Value function loss coefficient
        "max_grad_norm": 0.5,       # Gradient clipping
        "device": args.device,
        "verbose": 1
    }
    
    # Trading-specific adjustments for stochastic financial markets
    if "financial_adjustments" in args and args.financial_adjustments:
        # Increase batch size for more stable gradients in financial data
        ppo_params["batch_size"] = 512
        # Increase entropy coefficient for more exploration in volatile markets
        ppo_params["ent_coef"] = 0.02
        # Reduce clipping for more policy flexibility in dynamic markets
        ppo_params["clip_range"] = 0.1
    
    # Apply market-specific adjustments based on trading timeframe
    if "market_type" in args:
        if args.market_type == "short_term":  # Day trading (minutes to hours)
            # Use smaller gamma to focus more on immediate rewards
            ppo_params["gamma"] = 0.99
            # Increase entropy to adapt to rapidly changing market conditions
            ppo_params["ent_coef"] = 0.03
            # Smaller batch size for quicker updates
            ppo_params["n_steps"] = 1024
            # More epochs for thorough learning on smaller batches
            ppo_params["n_epochs"] = 12
            # Network architecture tuned for short-term patterns
            policy_kwargs["net_arch"] = {
                "pi": [128, 128, 64],  # Balanced network
                "vf": [128, 128, 64]
            }
            ppo_params["policy_kwargs"] = policy_kwargs
            
        elif args.market_type == "medium_term":  # Swing trading (days to weeks)
            # Higher gamma to consider more future rewards
            ppo_params["gamma"] = 0.997
            # Balanced entropy for exploration/exploitation
            ppo_params["ent_coef"] = 0.015
            # Larger batch size for more stable learning
            ppo_params["n_steps"] = 2048
            # Moderate number of epochs
            ppo_params["n_epochs"] = 8
            # Network architecture with deeper layers
            policy_kwargs["net_arch"] = {
                "pi": [256, 256, 128],  # Deeper network for more complex patterns
                "vf": [256, 256, 128]
            }
            ppo_params["policy_kwargs"] = policy_kwargs
            # More conservative clipping for stability
            ppo_params["clip_range"] = 0.15
            
        elif args.market_type == "long_term":  # Position trading (weeks to months)
            # Very high gamma for long-term reward consideration
            ppo_params["gamma"] = 0.999
            # Lower entropy to exploit learned strategies
            ppo_params["ent_coef"] = 0.005
            # Large batch size for stable, conservative updates
            ppo_params["n_steps"] = 4096
            # Fewer epochs to prevent overfitting
            ppo_params["n_epochs"] = 6
            # Network architecture with wider and deeper layers
            policy_kwargs["net_arch"] = {
                "pi": [384, 384, 256, 128],  # Wider and deeper for long-term patterns
                "vf": [384, 384, 256, 128]
            }
            ppo_params["policy_kwargs"] = policy_kwargs
            # Conservative clipping for stability
            ppo_params["clip_range"] = 0.1
            # Higher lambda for better long-term credit assignment
            ppo_params["gae_lambda"] = 0.98
            # Higher value function coefficient for accurate value estimation
            ppo_params["vf_coef"] = 0.7
    
    return ppo_params