"""
Black–Scholes option pricing utilities.

This module provides functions for pricing European options and computing
Greeks under the Black–Scholes framework. It is used as a baseline hedging
model throughout the project.
"""

import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Compute the Black-Scholes price of a call option
    
    Parameters:
    -----------
    S : float
        Current underlying asset price
    K : float
        Strike price of the option
    T : float
        Time to maturity in years
    r : float
        Continuously compounded risk-free interest rate
    sigma : float
        Annualized volatility of the underlying asset
    
    Returns:
    --------
    float : Theoretical Black–Scholes call option price
    """
    
    # intermediate calcs
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price


def black_scholes_delta(S, K, T, r, sigma):
    """
    Calculate the hedge ratio
    
    Parameters: Same as above
    
    Returns:
    --------
    float : Delta value between 0 and 1
    """
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    delta = norm.cdf(d1)
    
    return delta


def black_scholes_gamma(S, K, T, r, sigma):
    """
    Calculate Gamma
    
    Parameters: Same as above

    Returns:
    --------
    float : Gamma value
    """
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return gamma


def black_scholes_theta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Theta
    
    Parameters: Same as above 
        option_type: float
            whether the option is a call or a put
    
    Returns:
    --------
    float : Theta value (usually negative for long options)
    """
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
    # converting to a per-day value
    theta = theta / 365
    
    return theta

    