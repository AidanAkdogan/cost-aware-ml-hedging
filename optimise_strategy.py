"""
optimise_strategy.py

Bayesian optimization of two-stage execution parameters for cost-aware hedging

This script tunes two execution hyperparameters used by the learned two-stage
hedging controller:
- gate_threshold: defines the no-trade region (trade only if alpha >= threshold)
- convex_power: convexifies the executed alpha (alpha_exec = alpha ** convex_power)

The objective is to maximize a risk-adjusted performance score (Sharpe ratio),
optionally penalized by transaction costs and hedging error volatility, using Optuna
"""

import numpy as np
import pandas as pd
import pickle
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple

from black_scholes import black_scholes_call, black_scholes_delta, black_scholes_gamma, black_scholes_theta
from market_simulator import StockSimulator


class OptimizableBacktest:
    """
    Streamlined backtester for optimization.
    """
    
    def __init__(self, S0=100.0, K=105.0, T=30/252, r=0.05, sigma=0.2, 
                 transaction_cost=0.05, n_steps=30):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.transaction_cost = transaction_cost
        self.n_steps = n_steps
        self.dt = T / n_steps
        
        # fetch teh model
        with open('models/ml_hedge.pkl', 'rb') as f:
            model_package = pickle.load(f)
        self.model = model_package['model']
        self.feature_names = model_package['feature_names']
        print("The model has been loaded")
    
    def run_episode(self, gate_threshold: float, convex_power: float, seed: int) -> Dict:
        """
        Run single episode with given parameters.
        """
        
        stock_sim = StockSimulator(S0=self.S0, mu=self.r, sigma=self.sigma, dt=self.dt, seed=seed)
        S = stock_sim.reset(S0=self.S0)
        
        hedge_ratio = 0.0
        cumulative_cost = 0.0
        pnl_history = []
        num_trades = 0
        
        for step in range(self.n_steps):
            time_remaining = (self.n_steps - step) * self.dt
            
            if time_remaining > 1e-6:
                V = black_scholes_call(S, self.K, time_remaining, self.r, self.sigma)
                delta = black_scholes_delta(S, self.K, time_remaining, self.r, self.sigma)
                gamma = black_scholes_gamma(S, self.K, time_remaining, self.r, self.sigma)
                theta = black_scholes_theta(S, self.K, time_remaining, self.r, self.sigma, 'call')
            else:
                V = max(S - self.K, 0)
                delta = 1.0 if S > self.K else 0.0
                gamma = 0.0
                theta = 0.0
            
            delta_change = abs(delta - hedge_ratio)
            
            if step > 0:
                price_move = (S - S_prev) / S_prev
            else:
                price_move = 0.0
            
            # construct the state
            state = {
                'moneyness': S / self.K,
                'time_fraction': time_remaining / self.T,
                'option_value': V,
                'current_hedge': hedge_ratio,
                'bs_delta': delta,
                'delta_change': delta_change,
                'gamma': gamma,
                'theta': theta,
                'gamma_normalized': gamma * S,
                'gamma_squared': gamma ** 2,
                'trade_cost': delta_change * self.transaction_cost,
                'cost_to_value_ratio': (delta_change * self.transaction_cost) / (V + 1e-6),
                'time_to_expiry_days': time_remaining * 252,
                'is_near_expiry': 1 if time_remaining < 5/252 else 0,
                'is_atm': 1 if 0.95 < S / self.K < 1.05 else 0,
                'is_high_gamma': 1 if gamma > 0.01 else 0,
                'price_move': price_move,
                'abs_moneyness_deviation': abs(S / self.K - 1.0)
            }
            
            # Stage 1 (raw prediction)
            X = pd.DataFrame([state])[self.feature_names]
            alpha_raw = self.model.predict(X)[0]
            alpha_raw = np.clip(alpha_raw, 0.0, 1.0)
            
            # Stage 2 (emforce the trade gate)
            if alpha_raw < gate_threshold:
                new_hedge = hedge_ratio
            else:
                # convexify
                alpha_exec = np.sign(alpha_raw) * (np.abs(alpha_raw) ** convex_power)
                alpha_exec = np.clip(alpha_exec, 0.0, 1.0)
                
                # force a full rebalance if extreme 
                if delta_change > 0.12 and gamma > 0.01:
                    alpha_exec = 1.0
                
                new_hedge = hedge_ratio + alpha_exec * (delta - hedge_ratio)
                new_hedge = np.clip(new_hedge, 0.0, 1.0)
            
            # execute trade
            if abs(new_hedge - hedge_ratio) > 1e-6:
                trade_cost = abs(new_hedge - hedge_ratio) * self.transaction_cost
                cumulative_cost += trade_cost
                hedge_ratio = new_hedge
                num_trades += 1
            else:
                trade_cost = 0.0
            
            # market movement
            S_prev = S
            V_prev = V
            S = stock_sim.step()
            
            time_remaining_after = (self.n_steps - step - 1) * self.dt
            if time_remaining_after > 0:
                V_after = black_scholes_call(S, self.K, time_remaining_after, self.r, self.sigma)
            else:
                V_after = max(S - self.K, 0)
            
            # P&L
            option_pnl = -(V_after - V_prev)
            stock_pnl = hedge_ratio * (S - S_prev)
            step_pnl = option_pnl + stock_pnl - trade_cost
            pnl_history.append(step_pnl)
        
        return {
            'final_pnl': sum(pnl_history),
            'pnl_std': np.std(pnl_history),
            'total_cost': cumulative_cost,
            'num_trades': num_trades
        }
    
    def evaluate_params(
        self, 
        gate_threshold: float, 
        convex_power: float,
        n_episodes: int = 200,
        seed_blocks: list = None
    ) -> Dict:
        """
        Evaluate parameter pair across multiple seed blocks
        
        Using multiple seed blocks reduces noise in optimization
        """
        
        if seed_blocks is None:
            # using 3 blocks each of 200 episodes
            seed_blocks = [
                (50000, 50200),
                (60000, 60200),
                (70000, 70200)
            ]
        
        all_results = []
        
        for seed_start, seed_end in seed_blocks:
            block_results = []
            
            for episode in range(seed_start, seed_end):
                result = self.run_episode(gate_threshold, convex_power, seed=episode)
                block_results.append(result)
            
            # aggregate current block
            block_df = pd.DataFrame(block_results)
            sharpe = block_df['final_pnl'].mean() / (block_df['final_pnl'].std() + 1e-6)
            
            all_results.append({
                'sharpe': sharpe,
                'mean_cost': block_df['total_cost'].mean(),
                'mean_pnl_std': block_df['pnl_std'].mean(),
                'avg_trades': block_df['num_trades'].mean(),
                'final_pnl_mean': block_df['final_pnl'].mean(),
                'final_pnl_std': block_df['final_pnl'].std()
            })
        
        # avg across blocks
        avg_sharpe = np.mean([r['sharpe'] for r in all_results])
        avg_cost = np.mean([r['mean_cost'] for r in all_results])
        avg_pnl_std = np.mean([r['mean_pnl_std'] for r in all_results])
        avg_trades = np.mean([r['avg_trades'] for r in all_results])
        
        return {
            'sharpe': avg_sharpe,
            'mean_cost': avg_cost,
            'mean_pnl_std': avg_pnl_std,
            'avg_trades': avg_trades
        }


class BayesianOptimizer:
    """
    Bayesian optimization for strategy parameters.
    """
    
    def __init__(
        self,
        backtester: OptimizableBacktest,
        lambda_cost: float = 0.0,      # penalty for high cost
        lambda_vol: float = 0.0,       # penalty for high volatility
        baseline_sharpe: float = None  # optional, optimize vs baseline
    ):
        self.backtester = backtester
        self.lambda_cost = lambda_cost
        self.lambda_vol = lambda_vol
        self.baseline_sharpe = baseline_sharpe
        
        self.trial_history = []
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.
        
        Returns a score to MAXIMIZE.
        """
        
        # sample parameters
        gate_threshold = trial.suggest_float('gate_threshold', 0.10, 0.40, step=0.01)
        convex_power = trial.suggest_float('convex_power', 1.0, 3.0, step=0.05)
        
        # evaluate
        metrics = self.backtester.evaluate_params(
            gate_threshold=gate_threshold,
            convex_power=convex_power,
            seed_blocks=[
                (50000, 50200),
                (60000, 60200),
                (70000, 70200)
            ]
        )
        
        # build objective
        sharpe = metrics['sharpe']
        cost = metrics['mean_cost']
        vol = metrics['mean_pnl_std']
        
        # simple Sharpe maximization
        if self.baseline_sharpe is None:
            score = sharpe - self.lambda_cost * cost - self.lambda_vol * vol
        
        # beat baseline with constraints
        else:
            sharpe_improvement = sharpe - self.baseline_sharpe
            score = sharpe_improvement - self.lambda_cost * cost - self.lambda_vol * vol
        
        # store for analysis
        self.trial_history.append({
            'trial': trial.number,
            'gate_threshold': gate_threshold,
            'convex_power': convex_power,
            'sharpe': sharpe,
            'cost': cost,
            'vol': vol,
            'trades': metrics['avg_trades'],
            'score': score
        })
        
        # report intermediate value (pruning)
        trial.report(score, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
    
    def optimize(
        self, 
        n_trials: int = 50,
        show_progress: bool = True
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run Bayesian optimization.
        
        Returns:
        --------
        best_params : dict
            Optimal gate_threshold and convex_power
        history_df : pd.DataFrame
            Full trial history
        """
        
        print("=" * 70)
        print("RUNNING BAYESIAN OPTIMIZATION")
        print("=" * 70)
        print(f"\nOptimization setup:")
        print(f"  Trials: {n_trials}")
        print(f"  Parameter ranges:")
        print(f"    gate_threshold: [0.10, 0.40]")
        print(f"    convex_power: [1.0, 3.0]")
        print(f"\n  Objective components:")
        print(f"    Sharpe (primary)")
        print(f"    - {self.lambda_cost} × cost")
        print(f"    - {self.lambda_vol} × volatility")
        
        if self.baseline_sharpe:
            print(f"\n  Baseline Sharpe: {self.baseline_sharpe:.4f}")
        
        # create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
    
        if show_progress:
            study.optimize(
                self.objective, 
                n_trials=n_trials,
                show_progress_bar=True
            )
        else:
            for trial_num in tqdm(range(n_trials), desc="Optimizing"):
                study.optimize(self.objective, n_trials=1)
        
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"\nBest parameters:")
        print(f"   gate_threshold: {best_params['gate_threshold']:.3f}")
        print(f"   convex_power: {best_params['convex_power']:.3f}")
        print(f"\n📊 Best score: {best_value:.6f}")
        
        best_metrics = self.backtester.evaluate_params(
            gate_threshold=best_params['gate_threshold'],
            convex_power=best_params['convex_power']
        )
        
        print(f"\nBest configuration metrics:")
        print(f"   Sharpe: {best_metrics['sharpe']:.4f}")
        print(f"   Mean Cost: ${best_metrics['mean_cost']:.4f}")
        print(f"   P&L Std: ${best_metrics['mean_pnl_std']:.4f}")
        print(f"   Avg Trades: {best_metrics['avg_trades']:.1f}")
        
        history_df = pd.DataFrame(self.trial_history)
        
        return {
            'gate_threshold': best_params['gate_threshold'],
            'convex_power': best_params['convex_power'],
            'best_score': best_value,
            'best_metrics': best_metrics,
            'study': study
        }, history_df
    
    def visualize_results(
        self, 
        study: optuna.Study, 
        history_df: pd.DataFrame,
        save_path: str = 'results/optimization_results.png'
    ):
        """
        Create visualization of optimization results.
        """
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Create subdirectory for individual plots
        individual_dir = os.path.join(os.path.dirname(save_path), 'individual_plots')
        os.makedirs(individual_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Optimization history
        trials = history_df['trial'].values
        scores = history_df['score'].values
        best_scores = np.maximum.accumulate(scores)
        
        axes[0, 0].plot(trials, scores, 'o-', alpha=0.6, label='Trial score')
        axes[0, 0].plot(trials, best_scores, 'r-', linewidth=2, label='Best so far')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Optimization Progress', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter importance
        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(study)
            
            params = list(importances.keys())
            values = list(importances.values())
            
            axes[0, 1].barh(params, values, color='steelblue')
            axes[0, 1].set_xlabel('Importance')
            axes[0, 1].set_title('Parameter Importance', fontweight='bold')
        except:
            axes[0, 1].text(0.5, 0.5, 'Importance\nN/A', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Cost vs Sharpe
        axes[0, 2].scatter(history_df['cost'], history_df['sharpe'], 
                          c=history_df['score'], cmap='viridis', alpha=0.6)
        axes[0, 2].set_xlabel('Mean Cost ($)')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].set_title('Cost-Sharpe Trade-off', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2])
        cbar.set_label('Score')
        
        # Gate threshold vs Score
        axes[1, 0].scatter(history_df['gate_threshold'], history_df['score'], 
                          alpha=0.6, c='blue')
        axes[1, 0].set_xlabel('Gate Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Gate Threshold Effect', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Convex power vs Score
        axes[1, 1].scatter(history_df['convex_power'], history_df['score'], 
                          alpha=0.6, c='green')
        axes[1, 1].set_xlabel('Convex Power')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Convexity Effect', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 2D parameter space heatmap
        from scipy.interpolate import griddata
        
        gate = history_df['gate_threshold'].values
        power = history_df['convex_power'].values
        score = history_df['score'].values
        
        gate_grid = np.linspace(gate.min(), gate.max(), 50)
        power_grid = np.linspace(power.min(), power.max(), 50)
        gate_mesh, power_mesh = np.meshgrid(gate_grid, power_grid)
        
        score_mesh = griddata((gate, power), score, (gate_mesh, power_mesh), method='cubic')
        
        im = axes[1, 2].contourf(gate_mesh, power_mesh, score_mesh, levels=20, cmap='RdYlGn')
        axes[1, 2].scatter(gate, power, c='black', s=20, alpha=0.3)
        
        # Mark best
        best_idx = history_df['score'].idxmax()
        best_gate = history_df.loc[best_idx, 'gate_threshold']
        best_power = history_df.loc[best_idx, 'convex_power']
        axes[1, 2].scatter([best_gate], [best_power], c='red', s=200, marker='*', 
                          edgecolors='black', linewidths=2, label='Best')
        
        axes[1, 2].set_xlabel('Gate Threshold')
        axes[1, 2].set_ylabel('Convex Power')
        axes[1, 2].set_title('Parameter Space Heatmap', fontweight='bold')
        axes[1, 2].legend()
        plt.colorbar(im, ax=axes[1, 2], label='Score')
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # INDIVIDUAL PLOTS
        
        # Plot 1: Optimization Progress
        fig1 = plt.figure(figsize=(10, 6))
        plt.plot(trials, scores, 'o-', alpha=0.6, label='Trial score', markersize=4)
        plt.plot(trials, best_scores, 'r-', linewidth=2.5, label='Best so far')
        plt.xlabel('Trial Number', fontsize=13)
        plt.ylabel('Objective Score', fontsize=13)
        plt.title('Bayesian Optimization Progress', fontweight='bold', fontsize=15)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path1 = os.path.join(individual_dir, 'opt_01_progress.png')
        plt.savefig(path1, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path1}")
        
        # Plot 2: Parameter Importance
        fig2 = plt.figure(figsize=(8, 6))
        try:
            from optuna.importance import get_param_importances
            importances = get_param_importances(study)
            params = list(importances.keys())
            values = list(importances.values())
            plt.barh(params, values, color='steelblue', alpha=0.8)
            plt.xlabel('Importance Score', fontsize=13)
            plt.ylabel('Parameter', fontsize=13)
            plt.title('Parameter Importance Analysis', fontweight='bold', fontsize=15)
            plt.tight_layout()
            path2 = os.path.join(individual_dir, 'opt_02_importance.png')
            plt.savefig(path2, dpi=150, bbox_inches='tight')
            print(f"Saved: {path2}")
        except:
            print(f"Skipping parameter importance (not available)")
        plt.close()
        
        # Plot 3: Cost-Sharpe Trade-off
        fig3 = plt.figure(figsize=(10, 7))
        scatter = plt.scatter(history_df['cost'], history_df['sharpe'], 
                            c=history_df['score'], cmap='viridis', alpha=0.7, s=80, edgecolors='black', linewidths=0.5)
        plt.xlabel('Mean Transaction Cost ($)', fontsize=13)
        plt.ylabel('Sharpe Ratio', fontsize=13)
        plt.title('Cost-Performance Trade-off', fontweight='bold', fontsize=15)
        plt.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Objective Score', fontsize=12)
        plt.tight_layout()
        path3 = os.path.join(individual_dir, 'opt_03_cost_sharpe.png')
        plt.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path3}")
        
        # Plot 4: Gate Threshold Effect
        fig4 = plt.figure(figsize=(10, 6))
        plt.scatter(history_df['gate_threshold'], history_df['score'], 
                   alpha=0.7, c='blue', s=80, edgecolors='black', linewidths=0.5)
        plt.xlabel('Gate Threshold', fontsize=13)
        plt.ylabel('Objective Score', fontsize=13)
        plt.title('Effect of Gate Threshold on Performance', fontweight='bold', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path4 = os.path.join(individual_dir, 'opt_04_gate_effect.png')
        plt.savefig(path4, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path4}")
        
        # Plot 5: Convex Power Effect
        fig5 = plt.figure(figsize=(10, 6))
        plt.scatter(history_df['convex_power'], history_df['score'], 
                   alpha=0.7, c='green', s=80, edgecolors='black', linewidths=0.5)
        plt.xlabel('Convex Power', fontsize=13)
        plt.ylabel('Objective Score', fontsize=13)
        plt.title('Effect of Convexity on Performance', fontweight='bold', fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path5 = os.path.join(individual_dir, 'opt_05_power_effect.png')
        plt.savefig(path5, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path5}")
        
        # Plot 6: 2D Parameter Space
        fig6 = plt.figure(figsize=(12, 8))
        im = plt.contourf(gate_mesh, power_mesh, score_mesh, levels=20, cmap='RdYlGn', alpha=0.9)
        plt.scatter(gate, power, c='black', s=30, alpha=0.4, label='Trials')
        plt.scatter([best_gate], [best_power], c='red', s=300, marker='*', 
                   edgecolors='black', linewidths=2, label='Optimal', zorder=5)
        plt.xlabel('Gate Threshold', fontsize=13)
        plt.ylabel('Convex Power', fontsize=13)
        plt.title('Parameter Space Landscape', fontweight='bold', fontsize=15)
        plt.legend(fontsize=11)
        cbar = plt.colorbar(im)
        cbar.set_label('Objective Score', fontsize=12)
        plt.tight_layout()
        path6 = os.path.join(individual_dir, 'opt_06_parameter_space.png')
        plt.savefig(path6, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path6}")
        


def main():
    """
    Full optimization pipeline.
    """
    
    print("=" * 70)
    print("BAYESIAN PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Create backtester
    backtester = OptimizableBacktest()
    
    # Get baseline (Black-Scholes) for reference
    print("\nComputing baseline (Black-Scholes)...")
    bs_results = []
    for seed in range(50000, 50100):
        stock_sim = StockSimulator(S0=100.0, mu=0.05, sigma=0.2, dt=1/252, seed=seed)
        S = stock_sim.reset(S0=100.0)
        hedge = 0.0
        pnl = []
        
        for _ in range(30):
            T_rem = max(0.001, (30 - _) / 252)
            delta = black_scholes_delta(S, 105.0, T_rem, 0.05, 0.2)
            S_prev = S
            S = stock_sim.step()
            V_prev = black_scholes_call(S_prev, 105.0, T_rem, 0.05, 0.2)
            V = black_scholes_call(S, 105.0, max(0.001, T_rem - 1/252), 0.05, 0.2)
            pnl.append(-(V - V_prev) + delta * (S - S_prev))
            hedge = delta
        
        bs_results.append(sum(pnl))
    
    baseline_sharpe = np.mean(bs_results) / (np.std(bs_results) + 1e-6)
    print(f"Baseline Sharpe: {baseline_sharpe:.4f}\n")
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        backtester=backtester,
        lambda_cost=0.0,      # Start with pure Sharpe optimization
        lambda_vol=0.0,
        baseline_sharpe=baseline_sharpe
    )
    
    # Run optimisation
    result, history = optimizer.optimize(
        n_trials=50,
        show_progress=True
    )
    
    # visualise
    optimizer.visualize_results(
        study=result['study'],
        history_df=history
    )
    
    
    history.to_csv('results/optimization_history.csv', index=False)
    
    with open('results/best_params.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Saved:")
    print(f"   - results/optimization_history.csv")
    print(f"   - results/best_params.pkl")
    print(f"   - results/optimization_results.png")
    
    # Compare with current defaults
    print("\n" + "=" * 70)
    print("COMPARISON WITH DEFAULTS")
    print("=" * 70)
    
    default_gate = 0.25
    default_power = 1.5
    
    default_metrics = backtester.evaluate_params(default_gate, default_power)
    
    print(f"\nDefault params (gate=0.25, power=1.5):")
    print(f"   Sharpe: {default_metrics['sharpe']:.4f}")
    print(f"   Cost: ${default_metrics['mean_cost']:.4f}")
    print(f"   Trades: {default_metrics['avg_trades']:.1f}")
    
    print(f"\nOptimized params (gate={result['gate_threshold']:.3f}, power={result['convex_power']:.3f}):")
    print(f"   Sharpe: {result['best_metrics']['sharpe']:.4f}")
    print(f"   Cost: ${result['best_metrics']['mean_cost']:.4f}")
    print(f"   Trades: {result['best_metrics']['avg_trades']:.1f}")
    
    sharpe_improvement = ((result['best_metrics']['sharpe'] / default_metrics['sharpe']) - 1) * 100
    cost_change = ((result['best_metrics']['mean_cost'] / default_metrics['mean_cost']) - 1) * 100
    
    print(f"\nImprovement:")
    print(f"   Sharpe: {sharpe_improvement:+.1f}%")
    print(f"   Cost: {cost_change:+.1f}%")

if __name__ == "__main__":
    main()