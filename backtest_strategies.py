import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict

from black_scholes import black_scholes_call, black_scholes_delta, black_scholes_gamma, black_scholes_theta
from market_simulator import StockSimulator


class TwoStageBacktest:
    """
    Backtest hedging strategies on simulated price paths

    The backtester runs multiple Monte Carlo episodes, applies a user-provided
    strategy at each time step, and records:
    - cumulative transaction costs,
    - step-wise and terminal hedging P&L,
    - trade counts and hedging error proxies
    """
    
    def __init__(self, S0=100.0, K=105.0, T=30/252, r=0.05, sigma=0.2, transaction_cost=0.05, n_steps=30):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.transaction_cost = transaction_cost
        self.n_steps = n_steps
        self.dt = T / n_steps
    
    def run_episode(self, strategy_name, strategy_func, seed=None) -> Dict:
        """
        Run a single simulated episode for a specific hedging strategy

        At each step, the method will:
        1) compute the option state and Greeks,
        2) query the strategy for (alpha, new_hedge),
        3) apply transaction costs when the hedge changes,
        4) advance the simulator by one step,
        5) record step P&L for a short option hedger

        Parameters
        ----------
        strategy_name : str
            The label used in the results table
        strategy_func : Callable[[dict], Tuple[float, float]]
            Strategy callback that maps a state dict to (action_alpha, new_hedge)
        seed : int, optional
            Seed for reproducibility

        Returns
        -------
        dict
            Episode-level results including final P&L, P&L std, total cost, and trade count
        """
        
        stock_sim = StockSimulator(S0=self.S0, mu=self.r, sigma=self.sigma, dt=self.dt, seed=seed)
        S = stock_sim.reset(S0=self.S0)
        
        hedge_ratio = 0.0
        cumulative_cost = 0.0
        pnl_history = []
        hedge_history = []
        trade_history = []
        
        option_value = black_scholes_call(S, self.K, self.T, self.r, self.sigma)
        
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
            
            if step > 0 and hasattr(self, 'S_prev'):
                price_move = (S - self.S_prev) / self.S_prev
            else:
                price_move = 0.0
            
            state = {
                'stock_price': S, 'moneyness': S / self.K, 'option_value': V,
                'time_remaining': time_remaining, 'time_fraction': time_remaining / self.T,
                'current_hedge': hedge_ratio, 'bs_delta': delta, 'delta_change': delta_change,
                'gamma': gamma, 'theta': theta, 'gamma_normalized': gamma * S,
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
            
            action_alpha, new_hedge = strategy_func(state)
            
            if abs(new_hedge - hedge_ratio) > 1e-6:
                trade_cost = abs(new_hedge - hedge_ratio) * self.transaction_cost
                cumulative_cost += trade_cost
                hedge_ratio = new_hedge
                trade_history.append(1)
            else:
                trade_cost = 0.0
                trade_history.append(0)
            
            hedge_history.append(hedge_ratio)
            
            # market movement
            self.S_prev = S
            S_prev = S
            V_prev = V
            
            S = stock_sim.step()
            
            time_remaining_after = (self.n_steps - step - 1) * self.dt
            if time_remaining_after > 0:
                option_value = black_scholes_call(S, self.K, time_remaining_after, self.r, self.sigma)
            else:
                option_value = max(S - self.K, 0)
            
            # P&L
            option_pnl = -(option_value - V_prev)
            stock_pnl = hedge_ratio * (S - S_prev)
            step_pnl = option_pnl + stock_pnl - trade_cost
            
            pnl_history.append(step_pnl)
        
        return {
            'strategy': strategy_name,
            'final_pnl': sum(pnl_history),
            'pnl_std': np.std(pnl_history),
            'total_cost': cumulative_cost,
            'num_trades': sum(trade_history),
            'pnl_history': pnl_history,
            'hedge_history': hedge_history,
            'trade_history': trade_history
        }
    
    def backtest(self, strategies, n_episodes=1000, start_seed=50000) -> pd.DataFrame:
        """
        Runs a multi-strategy Monte Carlo backtest

        Parameters
        ----------
        strategies : dict[str, Callable]
            Mapping from strategy name to strategy callable
        n_episodes : int
            Number of episodes per strategy
        start_seed : int
            Starting seed for the episode block

        Returns
        -------
        pandas.DataFrame
            Long-form results with one row per (episode, strategy)
        """
        
        
        all_results = []
        
        for name, func in strategies.items():
            print(f"\nTesting: {name}")
            
            for episode in tqdm(range(n_episodes), desc=f"  {name}", ncols=80):
                try:
                    result = self.run_episode(name, func, seed=start_seed + episode)
                    all_results.append(result)
                except Exception as e:
                    print(f"\n⚠️  Error: {e}")
                    continue
        
        return pd.DataFrame(all_results)
    
    def analyze_results(self, results_df) -> pd.DataFrame:
        """
        Aggregate episode-level results into a strategy summary table

        Computes mean/std of final P&L, mean P&L std, mean transaction cost, average number of trades, and Sharpe ratio

        Parameters
        ----------
        results_df : pandas.DataFrame
            Output of backtest(...)

        Returns
        -------
        pandas.DataFrame
            Each strategy's summary metrics
        """
        
        summary = results_df.groupby('strategy').agg({
            'final_pnl': ['mean', 'std'],
            'pnl_std': 'mean',
            'total_cost': 'mean',
            'num_trades': 'mean'
        }).round(4)
        
        summary.columns = ['Mean Final P&L', 'P&L Volatility', 'Mean P&L Std', 'Mean Cost', 'Avg Trades']
        
        for strategy in summary.index:
            data = results_df[results_df['strategy'] == strategy]
            sharpe = data['final_pnl'].mean() / (data['final_pnl'].std() + 1e-6)
            summary.loc[strategy, 'Sharpe Ratio'] = sharpe
        
        return summary


class BlackScholesStrategy:
    """
    Baseline strategy: rebalance fully to the Black–Scholes delta each step, assumes frictionless market

    Returns (alpha=1.0, new_hedge=bs_delta) at every decision point in time
    """
    def __call__(self, state):
        return 1.0, state['bs_delta']


class MLTwoStageStrategy:
    """
    Two-stage learned hedging controller

    Stage 1: trade gate
        If the predicted alpha is below `gate_threshold`, do not bother trading

    Stage 2: sizing it up
        If trading should occur, execute a convexified adjustment alpha_exec = alpha^p in order to sharpen decisions toward 0/1

    """
    
    def __init__(
        self, 
        model_path='models/ml_hedge.pkl',
        gate_threshold=0.25,  
        convex_power=1.5
    ):
        
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        self.model = model_package['model']
        self.feature_names = model_package['feature_names']
        self.gate_threshold = gate_threshold
        self.convex_power = convex_power
    
    def __call__(self, state):

        features_dict = {feat: state[feat] for feat in self.feature_names}
        features_df = pd.DataFrame([features_dict])
        
        # obtaining alfa hat
        alpha_raw = self.model.predict(features_df)[0]
        alpha_raw = np.clip(alpha_raw, 0.0, 1.0)
        
        
        # enforce the gating system (1)        
        if alpha_raw < self.gate_threshold:
            return 0.0, state['current_hedge']
        
        # quantify a viable trade (2)
        
        # we push toward extremes: α_exec = α^power
        # This creates: small α → smaller, large α → larger
        alpha_exec = np.sign(alpha_raw) * (np.abs(alpha_raw) ** self.convex_power)
        alpha_exec = np.clip(alpha_exec, 0.0, 1.0)
        
        # additionally, force a full rebalancing if large delta change + high gamma
        if state['delta_change'] > 0.12 and state['gamma'] > 0.01:
            alpha_exec = 1.0
        
        # calculaye the new hedge
        current_hedge = state['current_hedge']
        bs_delta = state['bs_delta']
        new_hedge = current_hedge + alpha_exec * (bs_delta - current_hedge)
        new_hedge = np.clip(new_hedge, 0.0, 1.0)
        
        return alpha_exec, new_hedge


class SimpleRulesStrategy:
    """
    Delta-threshold strategy.

    Rebalance fully to Black–Scholes delta only when the absolute delta gap
    exceeds a fixed threshold, otherwise just maintain the existing hedge as is
    """

    def __init__(self, threshold=0.05):
        self.threshold = threshold
    
    def __call__(self, state):
        if state['delta_change'] > self.threshold:
            return 1.0, state['bs_delta']
        else:
            return 0.0, state['current_hedge']


def create_plots(results_df, summary, save_path='results/backtest_results.png', individual_dir = 'results/individual_plots/'):
    """Create diagnostic plots for strategy comparison."""
    
    import os

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    strategies = results_df['strategy'].unique()
    colors = {'Black-Scholes': 'red', 'ML-TwoStage': 'blue', 'Simple Rules': 'green'}
    
    # Plots (same structure as before)
    ax1 = fig.add_subplot(gs[0, 0])
    for strategy in strategies:
        data = results_df[results_df['strategy'] == strategy]['total_cost']
        ax1.hist(data, bins=50, alpha=0.6, label=strategy, color=colors.get(strategy, 'gray'))
    ax1.set_xlabel('Cost ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Transaction Cost', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    for strategy in strategies:
        data = results_df[results_df['strategy'] == strategy]['pnl_std']
        ax2.hist(data, bins=50, alpha=0.6, label=strategy, color=colors.get(strategy, 'gray'))
    ax2.set_xlabel('P&L Std ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Hedging Quality', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    costs = summary['Mean Cost'].values
    x_pos = np.arange(len(strategies))
    ax3.bar(x_pos, costs, color=[colors.get(s, 'gray') for s in strategies])
    ax3.set_ylabel('Mean Cost ($)')
    ax3.set_title('Average Cost', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(strategies, rotation=15, ha='right')
    
    ax4 = fig.add_subplot(gs[1, 0])
    for strategy in strategies:
        data = results_df[results_df['strategy'] == strategy]
        ax4.scatter(data['total_cost'], data['pnl_std'], alpha=0.3, label=strategy, color=colors.get(strategy, 'gray'))
    ax4.set_xlabel('Cost ($)')
    ax4.set_ylabel('P&L Std ($)')
    ax4.set_title('Cost-Quality Trade-off', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Summary table
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    bs_cost = summary.loc['Black-Scholes', 'Mean Cost']
    bs_std = summary.loc['Black-Scholes', 'Mean P&L Std']
    
    summary_text = "RESULTS SUMMARY\n" + "="*50 + "\n\n"
    
    for strategy in strategies:
        cost = summary.loc[strategy, 'Mean Cost']
        std = summary.loc[strategy, 'Mean P&L Std']
        trades = summary.loc[strategy, 'Avg Trades']
        sharpe = summary.loc[strategy, 'Sharpe Ratio']
        
        if strategy != 'Black-Scholes':
            cost_improvement = (1 - cost/bs_cost) * 100
            std_change = ((std/bs_std) - 1) * 100
            
            summary_text += f"{strategy}:\n"
            summary_text += f"  Cost: ${cost:.4f} ({cost_improvement:+.1f}%)\n"
            summary_text += f"  Std: ${std:.4f} ({std_change:+.1f}%)\n"
            summary_text += f"  Trades: {trades:.1f}\n"
            summary_text += f"  Sharpe: {sharpe:.3f}\n\n"
        else:
            summary_text += f"{strategy}:\n"
            summary_text += f"  Cost: ${cost:.4f}\n"
            summary_text += f"  Std: ${std:.4f}\n"
            summary_text += f"  Trades: {trades:.1f}\n"
            summary_text += f"  Sharpe: {sharpe:.3f}\n\n"
    
    ax5.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax5.transAxes)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n\nComposite visualisation saved : {save_path}")
    plt.close()

    # INDIVIDUAL PLOTS
    
    # Plot 1: Transaction Cost Distribution
    fig1 = plt.figure(figsize=(10, 6))
    for strategy in strategies:
        data = results_df[results_df['strategy'] == strategy]['total_cost']
        plt.hist(data, bins=50, alpha=0.6, label=strategy, color=colors.get(strategy, 'gray'), edgecolor='black')
    plt.xlabel('Transaction Cost ($)', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.title('Distribution of Transaction Costs by Strategy', fontweight='bold', fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(individual_dir, 'backtest_01_cost_dist.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path1}")
    
    # Plot 2: Hedging Quality Distribution
    fig2 = plt.figure(figsize=(10, 6))
    for strategy in strategies:
        data = results_df[results_df['strategy'] == strategy]['pnl_std']
        plt.hist(data, bins=50, alpha=0.6, label=strategy, color=colors.get(strategy, 'gray'), edgecolor='black')
    plt.xlabel('P&L Standard Deviation ($)', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.title('Distribution of Hedging Quality by Strategy', fontweight='bold', fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(individual_dir, 'backtest_02_quality_dist.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")
    
    # Plot 3: Average Cost Comparison (Bar Chart)
    fig3 = plt.figure(figsize=(10, 6))
    costs = summary['Mean Cost'].values
    x_pos = np.arange(len(strategies))
    bars = plt.bar(x_pos, costs, color=[colors.get(s, 'gray') for s in strategies], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Mean Transaction Cost ($)', fontsize=13)
    plt.title('Average Cost by Strategy', fontweight='bold', fontsize=15)
    plt.xticks(x_pos, strategies, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    path3 = os.path.join(individual_dir, 'backtest_03_avg_cost.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path3}")
    
    # Plot 4: Cost-Quality Scatter
    fig4 = plt.figure(figsize=(10, 7))
    for strategy in strategies:
        data = results_df[results_df['strategy'] == strategy]
        plt.scatter(data['total_cost'], data['pnl_std'], alpha=0.4, 
                   label=strategy, color=colors.get(strategy, 'gray'), s=30, edgecolors='black', linewidths=0.5)
    plt.xlabel('Transaction Cost ($)', fontsize=13)
    plt.ylabel('P&L Standard Deviation ($)', fontsize=13)
    plt.title('Cost-Quality Trade-off Comparison', fontweight='bold', fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path4 = os.path.join(individual_dir, 'backtest_04_cost_quality.png')
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path4}")
    
    # Plot 5: Number of Trades Distribution
    fig5 = plt.figure(figsize=(10, 6))
    for strategy in strategies:
        data = results_df[results_df['strategy'] == strategy]['num_trades']
        plt.hist(data, bins=30, alpha=0.6, label=strategy, color=colors.get(strategy, 'gray'), edgecolor='black')
    plt.xlabel('Number of Trades', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.title('Trade Frequency Distribution by Strategy', fontweight='bold', fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path5 = os.path.join(individual_dir, 'backtest_05_trade_freq.png')
    plt.savefig(path5, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path5}")
    
    # Plot 6: Sharpe Ratio Comparison (Bar Chart)
    fig6 = plt.figure(figsize=(10, 6))
    sharpes = [summary.loc[s, 'Sharpe Ratio'] for s in strategies]
    x_pos = np.arange(len(strategies))
    bars = plt.bar(x_pos, sharpes, color=[colors.get(s, 'gray') for s in strategies], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Sharpe Ratio', fontsize=13)
    plt.title('Risk-Adjusted Performance Comparison', fontweight='bold', fontsize=15)
    plt.xticks(x_pos, strategies, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    path6 = os.path.join(individual_dir, 'backtest_06_sharpe.png')
    plt.savefig(path6, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path6}")


def main():
    """Backtesting pipeline."""
    
    backtester = TwoStageBacktest()
    
    strategies = {
        'Black-Scholes': BlackScholesStrategy(),
        'ML-TwoStage': MLTwoStageStrategy(
            gate_threshold=0.19,  # Tune this!
            convex_power=1.7      # Tune this!
        ),
        'Simple Rules': SimpleRulesStrategy(threshold=0.05)
    }
    
    print("\n💡 Testing TWO-STAGE CONTROL:")
    print("   Stage 1: Trade gate (α > 0.15)")
    print("   Stage 2: Convexify (α^1.5)")
    print("   + Force full rebalance if high gamma + large delta")
    
    results_df = backtester.backtest(strategies, n_episodes=1000, start_seed=50000)
    summary = backtester.analyze_results(results_df)
    
    print("\n" + "=" * 70)
    print(" DETAILED RESULTS")
    print("=" * 70)
    print("\n" + summary.to_string())
    
    print("\n" + "=" * 70)
    print(" COST SAVINGS ANALYSIS")
    print("=" * 70)
    
    bs_cost = summary.loc['Black-Scholes', 'Mean Cost']
    bs_std = summary.loc['Black-Scholes', 'Mean P&L Std']
    
    for strategy in summary.index:
        if strategy != 'Black-Scholes':
            cost = summary.loc[strategy, 'Mean Cost']
            std = summary.loc[strategy, 'Mean P&L Std']
            sharpe = summary.loc[strategy, 'Sharpe Ratio']
            bs_sharpe = summary.loc['Black-Scholes', 'Sharpe Ratio']
            
            cost_savings = (1 - cost/bs_cost) * 100
            std_change = ((std/bs_std) - 1) * 100
            sharpe_improvement = ((sharpe/bs_sharpe) - 1) * 100
            
            print(f"\n{strategy} vs Black-Scholes:")
            print(f"  Cost savings: {cost_savings:+.1f}%")
            print(f"  P&L std change: {std_change:+.1f}%")
            print(f"  Sharpe improvement: {sharpe_improvement:+.1f}%")
            
            
    
    create_plots(results_df, summary)
    
    results_df.to_csv('results/backtest_detailed.csv', index=False)
    summary.to_csv('results/backtest_summary.csv')


if __name__ == "__main__":
    main()