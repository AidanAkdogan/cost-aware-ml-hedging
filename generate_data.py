import numpy as np
import pickle
from typing import List, Dict, Tuple
from tqdm import tqdm

from black_scholes import black_scholes_call, black_scholes_delta, black_scholes_gamma, black_scholes_theta
from market_simulator import StockSimulator


class TwoStageDataGenerator:
    """
    Class that generates training data for a two-stage hedging control framework.

    The data generation process decomposes hedging decisions into:
    (i) a trade-gating decision-step that determines whether rebalancing should occur at all given market state and
    (ii) a continuous sizing decision that determines the fraction of the target hedge adjustment to execute
    
    Labels are constructed via counterfactual simulation to explicitly account
    for transaction costs and future hedging error.
    """
    
    def __init__(
        self,
        S0: float = 100.0,
        K: float = 105.0,
        T: float = 30/252,
        r: float = 0.05,
        sigma: float = 0.2,
        transaction_cost: float = 0.05,
        n_steps: int = 30,
        lookahead_window: int = 5,
        # Objective weights
        cost_penalty_lambda: float = 1.0,
        variance_penalty_eta: float = 0.8, 
        underhedge_penalty: float = 2.0     # heavily penalise under-hedging in high-gamma scenarios
    ):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.transaction_cost = transaction_cost
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.lookahead_window = lookahead_window
        self.cost_penalty_lambda = cost_penalty_lambda
        self.variance_penalty_eta = variance_penalty_eta
        self.underhedge_penalty = underhedge_penalty
        
        # potential action set -- simulating a continuos control problem without RL
        self.alpha_grid = np.arange(0.0, 1.05, 0.05)
    
    def generate_episode_with_path(self, seed: int) -> Tuple[List[float], List[Dict]]:
        """
        Simulate a single stock price pathway and the corresponding option states

        The method generates a GBM price path and computes the Black–Scholes option value and greeks at each time step.

        Parameters
        ----------
        seed : int
            Random seed just for reproducibility

        Returning
        -------
        stock_path : list of floats
            simulated stock prices over the episode
        states : list of dicts
            Per-step option state variables (price, delta, gamma, theta, etc.)
        """
        stock_sim = StockSimulator(
            S0=self.S0, 
            mu=self.r, 
            sigma=self.sigma,
            dt=self.dt, 
            seed=seed
        )
        
        stock_path = [stock_sim.reset(S0=self.S0)]

        for _ in range(self.n_steps - 1):
            stock_path.append(stock_sim.step())
        
        states = []

        for step, S in enumerate(stock_path):
            time_remaining = (self.n_steps - step) * self.dt
            
            if time_remaining > 1e-6:
                V = black_scholes_call(S, self.K, time_remaining, self.r, self.sigma)
                delta = black_scholes_delta(S, self.K, time_remaining, self.r, self.sigma)
                gamma = black_scholes_gamma(S, self.K, time_remaining, self.r, self.sigma)
                theta = black_scholes_theta(S, self.K, time_remaining, self.r, self.sigma, option_type='call')
            else:
                V = max(S - self.K, 0)
                delta = 1.0 if S > self.K else 0.0
                gamma = 0.0
                theta = 0.0
            
            states.append({
                'step': step,
                'stock_price': S,
                'option_value': V,
                'bs_delta': delta,
                'gamma': gamma,
                'theta': theta,
                'time_remaining': time_remaining,
                'moneyness': S / self.K
            })
        
        return stock_path, states
    
    def simulate_action_outcome(
        self,
        action_alpha: float,
        current_hedge: float,
        bs_delta: float,
        start_step: int,
        stock_path: List[float],
        states: List[Dict],
        current_gamma: float
    ) -> Tuple[float, float, float]:
        """
        Evaluating the effects of a potential hedge adjustment through counterfactual simulation

        This method simulates future P+L over a short timeframe assuming a partial hedge adjustment determined by the action_alpha

        An asymmetric penalty is applied to under-hedging in high-gamma moments and this was my key discovery to stop P&L blowing up!

        Parameters
        ----------
        action_alpha : float
            the fraction of the delta gap 0<action_alfa<1
        current_hedge : float
            The current hedge ratio
        bs_delta : float
            Black–Scholes calculated delta at the current step
        start_step : int
            Time index at which the action is evaluated
        stock_path : list of floats
            the simulated stock prices
        states : list of dict
            Option state variables along the pathway
        current_gamma : float
            Option gamma at the evaluation step

        Returns
        -------
        hedging_error : float
            the sum of squared P&L over the short term lookahead
        transaction_cost : float
            costs incurred by executing the hedge adjustment
        path_variance : float
            Variance of hedging P&L over the lookahead period
        underhedge_penalty : float
            Additional penalty for under-hedging in high-gamma moments **
        """
        
        new_hedge = current_hedge + action_alpha * (bs_delta - current_hedge)
        new_hedge = np.clip(new_hedge, 0.0, 1.0)
        
        delta_traded = abs(new_hedge - current_hedge)
        transaction_cost = delta_traded * self.transaction_cost
        
        hedge = new_hedge
        option_value = states[start_step]['option_value']
        S = stock_path[start_step]
        
        pnl_list = []
        end_step = min(start_step + self.lookahead_window, len(stock_path) - 1)
        
        for step in range(start_step + 1, end_step + 1):
            S_prev = S
            V_prev = option_value
            
            # precompute stock and option changes
            S = stock_path[step]
            option_value = states[step]['option_value']
            
            option_pnl = -(option_value - V_prev)
            stock_pnl = hedge * (S - S_prev)
            step_pnl = option_pnl + stock_pnl
            
            pnl_list.append(step_pnl)
        
        hedging_error = sum(p**2 for p in pnl_list)
        path_variance = np.var(pnl_list) if len(pnl_list) > 1 else 0.0
        
        # we do not want to under-hedge in a high-gamma moment
        hedge_gap = abs(new_hedge - bs_delta)
        underhedge_penalty = (hedge_gap ** 2) * (current_gamma ** 2) * self.underhedge_penalty
        
        # returning all of the costs of a hypothetical action
        return hedging_error, transaction_cost, path_variance, underhedge_penalty
    
    def find_optimal_alpha(
        self,
        current_hedge: float,
        bs_delta: float,
        start_step: int,
        stock_path: List[float],
        states: List[Dict],
        current_gamma: float
    ) -> float:
        """
        Selects the optimal hedge adjustment ratio through discrete search.

        The method evaluates a grid of candidate action_alpha values and selects
        the one minimizing a combined objective function consisting of:
        hedging error, transaction cost, P&L variance, and asymmetric under-hedging penalty.

        Parameters
        ----------
        current_hedge : float
            The current hedge ratio
        bs_delta : float
            Black–Scholes delta at our current step
        start_step : int
            Time index when the action is evaluated
        stock_path : list of float
            Simulated stock prices
        states : list of dict
            Option state variables along the path
        current_gamma : float
            Option gamma at the evaluation step

        Returns
        -------
        optimal_alpha : float
            adjustment fraction minimizing the objective function value
        """
        
        best_score = float('inf')
        best_alpha = 0.0
        
        for alpha in self.alpha_grid:
            hedging_error, cost, path_variance, underhedge_penalty = self.simulate_action_outcome(
                action_alpha=alpha,
                current_hedge=current_hedge,
                bs_delta=bs_delta,
                start_step=start_step,
                stock_path=stock_path,
                states=states,
                current_gamma=current_gamma
            )
            
            # the golden objective function
            score = (hedging_error + 
                    self.cost_penalty_lambda * cost + 
                    self.variance_penalty_eta * path_variance +
                    underhedge_penalty)
            
            if score < best_score:
                best_score = score
                best_alpha = alpha
        
        return best_alpha
    
    def create_training_examples_from_episode(
        self,
        stock_path: List[float],
        states: List[Dict]
    ) -> List[Dict]:
        """
        This method constructs supervised training examples from a single simulated episode

        For each eligible time step, the method will:
        - compute the optimal hedge adjustment via counterfactual evaluation (as above),
        - extract state features,
        - assign a continuous target alpha,
        - apply gamma-based sample weighting

        Parameters
        ----------
        stock_path : list of floats
            Our simmed stock prices
        states : list of dicts
            Option state variables for the episode

        Returns
        -------
        training_examples : list of dicts
            Feature dictionaries, target alpha values, and sample weights
        """
        
        training_examples = []
        current_hedge = 0.0
        
        max_step = len(stock_path) - self.lookahead_window - 1
        
        for step in range(max_step):
            state = states[step]
            bs_delta = state['bs_delta']
            gamma = state['gamma']
            
            # Finding the optimal alpha
            optimal_alpha = self.find_optimal_alpha(
                current_hedge=current_hedge,
                bs_delta=bs_delta,
                start_step=step,
                stock_path=stock_path,
                states=states,
                current_gamma=gamma
            )
            
            new_hedge = current_hedge + optimal_alpha * (bs_delta - current_hedge)
            new_hedge = np.clip(new_hedge, 0.0, 1.0)
            
            delta_change = abs(bs_delta - current_hedge)
            
            if step > 0:
                price_move = (stock_path[step] - stock_path[step-1]) / stock_path[step-1]
            else:
                price_move = 0.0
            
            features = {
                'moneyness': state['moneyness'],
                'time_fraction': state['time_remaining'] / self.T,
                'option_value': state['option_value'],
                'current_hedge': current_hedge,
                'bs_delta': bs_delta,
                'delta_change': delta_change,
                'gamma': gamma,
                'theta': state['theta'],
                'gamma_normalized': gamma * state['stock_price'],
                'gamma_squared': gamma ** 2,
                'trade_cost': delta_change * self.transaction_cost,
                'cost_to_value_ratio': (delta_change * self.transaction_cost) / (state['option_value'] + 1e-6),
                'time_to_expiry_days': state['time_remaining'] * 252,
                'is_near_expiry': 1 if state['time_remaining'] < 5/252 else 0,
                'is_atm': 1 if 0.95 < state['moneyness'] < 1.05 else 0,
                'is_high_gamma': 1 if gamma > np.percentile([s['gamma'] for s in states], 75) else 0,
                'price_move': price_move,
                'abs_moneyness_deviation': abs(state['moneyness'] - 1.0)
            }
            
            # sample weight to further stress high gamma environment in the learning process
            sample_weight = 1.0 + (gamma ** 2) * 10.0
            
            training_example = {
                'features': features,
                'target_alpha': optimal_alpha,
                'sample_weight': sample_weight,
                'delta_change': delta_change,
                'gamma': gamma
            }
            
            training_examples.append(training_example)
            current_hedge = new_hedge
        
        return training_examples
    
    def generate_dataset(
        self,
        n_episodes: int = 10000,
        save_path: str = 'data/training_data_two_stage.pkl'
    ) -> Dict:
        """
        Generates a full-fledged supervised dataset for two-stage hedging control.

        This method simulates n_episodes episodes, aggregates per-step training
        examples, reports relevant dataset statistics to the terminal, and serializes the dataset to disk

        Parameters
        ----------
        n_episodes : int
            the number of simulated episodes
        save_path : str
            Out-file path for the dataset

        Returns
        -------
        dataset : dict
            A dictionary containing training examples and generation parameters
        """
        
        all_training_examples = []
        
        for episode in tqdm(range(n_episodes), desc="Generating"):
            stock_path, states = self.generate_episode_with_path(seed=episode)
            examples = self.create_training_examples_from_episode(stock_path, states)
            all_training_examples.extend(examples)
        
        print(f"  Total examples: {len(all_training_examples):,}")
        
        alphas = [ex['target_alpha'] for ex in all_training_examples]
        weights = [ex['sample_weight'] for ex in all_training_examples]
        
        print(f"\nAlpha Distribution:")
        print(f"  Mean: {np.mean(alphas):.3f}")
        print(f"  Median: {np.median(alphas):.3f}")
        
        print(f"\nSample Weight Distribution:")
        print(f"  Mean: {np.mean(weights):.2f}")
        print(f"  Max: {np.max(weights):.2f}")
        
        dataset = {
            'training_examples': all_training_examples,
            'params': {
                'S0': self.S0,
                'K': self.K,
                'T': self.T,
                'sigma': self.sigma,
                'transaction_cost': self.transaction_cost,
                'n_episodes': n_episodes,
                'alpha_grid': self.alpha_grid.tolist(),
                'lookahead_window': self.lookahead_window,
                'cost_penalty_lambda': self.cost_penalty_lambda,
                'variance_penalty_eta': self.variance_penalty_eta,
                'underhedge_penalty': self.underhedge_penalty
            }
        }
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        return dataset


if __name__ == "__main__":
    generator = TwoStageDataGenerator(
        S0=100.0,
        K=105.0,
        T=30/252,
        r=0.05,
        sigma=0.2,
        transaction_cost=0.05,
        n_steps=30,
        lookahead_window=5,
        cost_penalty_lambda=1.0,
        variance_penalty_eta=0.8,
        underhedge_penalty=2.0
    )
    
    dataset = generator.generate_dataset(
        n_episodes=10000,
        save_path='data/training_data_two_stage.pkl'
    )
    