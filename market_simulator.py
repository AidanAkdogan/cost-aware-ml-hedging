import numpy as np
from typing import Tuple, List

class StockSimulator:
    """
    Market simulation utilities.

    This clkass provides tools for simulating underlying asset price dynamics
    under a geometric Brownian motion (GBM) model. It is used to generate
    synthetic price paths for evaluating hedging strategies under transaction
    costs in a controlled, reproducible setting.
    """


    def __init__(
        self,
        S0: float = 100.0,
        mu : float = 0.05,
        sigma: float = 0.2,
        dt : float = 1/252,
        seed: int = None
    ):
        """
        Instantiate the stock price sim.

        Parameters
        ----------
        S0 : float
            Initial asset price
        mu : float
            The drift term of the GBM process
        sigma : float
            Annualized volatility term
        dt : float
            Time step size in years
        seed : int, optional
            Random seed for reproducibility
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

        # current state
        self.S = S0
        self.time = 0

        self.rng = np.random.RandomState(seed)

    def reset(self, S0: float = None) -> float:
        """
        Reset to the initial state.
        
        Returns:
        --------
        float : the initial stock price
        """
        self.S = S0 if S0 is not None else self.S0
        self.time = 0
        return self.S


    def step(self) -> float:
        """
        Simulates one step in time

        Returns:
        --------
        New stock price as a float
        """
        
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma*np.sqrt(self.dt)*self.rng.randn()

        # updating stock prices in correspondance
        self.S = self.S *np.exp(drift + diffusion)
        self.time += self.dt

        return self.S

    def simulate_path(self, n_steps : int) -> np.ndarray:
        """
        Simulate price path with multiple steps

        Parameters:
        -----------
        n_steps : int
            Number of steps to be simulated

        Returns:
        --------
        np.ndarray : Array of stock prices : [S0, ... , Sn]
        """
        
        self.reset()

        path = [self.S]
        for _ in range(n_steps):
            path.append(self.step())

        return np.array(path)
