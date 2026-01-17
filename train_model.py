import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns


class TwoStageTrainer:
    """
    Train regressor with asymmetric loss and sample weighting
    """
    
    def __init__(self, data_path: str = 'data/training_data.pkl'):
        """
        Load training data and initialize trainer state.
        """
        with open(data_path, 'rb') as f:
            self.dataset = pickle.load(f)
        
        self.training_examples = self.dataset['training_examples']
        
        print(f"Loaded {len(self.training_examples):,} examples")
    
    def prepare_features(self):
        """Prepare features with sample weights"""
        
        feature_dicts = [ex['features'] for ex in self.training_examples]
        targets = [ex['target_alpha'] for ex in self.training_examples]
        sample_weights = [ex['sample_weight'] for ex in self.training_examples]
        
        X = pd.DataFrame(feature_dicts)
        y = np.array(targets)
        weights = np.array(sample_weights)
        
        self.feature_names = list(X.columns)
        
        print(f"\nFeatures ({len(self.feature_names)}):")
        for i, name in enumerate(self.feature_names):
            print(f"  {i+1:2d}. {name}")
        
        print(f"\nSample Weights:")
        print(f"  Mean: {np.mean(weights):.2f}")
        print(f"  Min: {np.min(weights):.2f}, Max: {np.max(weights):.2f}")
        print(f"  High-gamma states getting {np.max(weights)/np.min(weights):.1f}x more weight!")
        
        # test train split
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights,
            test_size=0.2, 
            random_state=42
        )
        
        print(f"\nDataset split:")
        print(f"  Training: {len(X_train):,} examples")
        print(f"  Test: {len(X_test):,} examples")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.w_train = w_train
        self.w_test = w_test
        
        return X_train, X_test, y_train, y_test, w_train, w_test
    
    def train_model(self):
        """
        Train with QUANTILE LOSS (70th percentile)
        
        This is to penalise under-hedging
        """
        
        print(f"\nTraining GradientBoosting with quantile loss...")
        print(f"   Quantile: 0.70 (penalizes under-prediction!)")
        
        self.model = GradientBoostingRegressor(
            n_estimators=200,          
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=100,
            min_samples_leaf=50,
            loss='quantile',           
            alpha=0.70,                
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        print(f"\n  Training on {len(self.X_train):,} examples...")
        
        # training with sample weights
        self.model.fit(self.X_train, self.y_train, sample_weight=self.w_train)
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the held-out test set

        Returns
        -------
        dict
            Metrics including accuracy, ROC-AUC, and a confusion matrix
        """
        
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        y_pred = self.model.predict(self.X_test)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        mse = mean_squared_error(self.y_test, y_pred, sample_weight=self.w_test)
        mae = mean_absolute_error(self.y_test, y_pred, sample_weight=self.w_test)
        r2 = r2_score(self.y_test, y_pred, sample_weight=self.w_test)
        
        print(f"\n1. Weighted Regression Metrics:")
        print(f"   MSE:  {mse:.6f}")
        print(f"   MAE:  {mae:.6f}")
        print(f"   R²:   {r2:.4f}")
        
        print(f"\n2. Prediction Distribution:")
        print(f"   Actual:    mean={np.mean(self.y_test):.3f}, std={np.std(self.y_test):.3f}")
        print(f"   Predicted: mean={np.mean(y_pred):.3f}, std={np.std(y_pred):.3f}")
        
        under_pred = np.sum(y_pred < self.y_test) / len(self.y_test) * 100
        print(f"\n3. Under-Prediction Rate:")
        print(f"   {under_pred:.1f}% (target is <40% with quantile=0.70)")
        
        print(f"\n4. Feature Importance:")
        print("-" * 70)
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop 10:")
        for i, idx in enumerate(indices[:10]):
            print(f"  {i+1:2d}. {self.feature_names[idx]:<30} {importances[idx]:.4f}")
        
        self.feature_importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred,
            'under_pred_rate': under_pred
        }
        
        return self.metrics
    
    def visualize_results(self, save_path='results/model_performance.png'):
        """
        Create and save diagnostic plots for model evaluation.

        Parameters
        ----------
        save_path : str
            Output path for the figure.
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        y_pred = self.metrics['y_pred']
        
        # Predicted vs Actual
        axes[0, 0].scatter(self.y_test, y_pred, alpha=0.3, s=10, c='blue')
        axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Alpha')
        axes[0, 0].set_ylabel('Predicted Alpha')
        axes[0, 0].set_title('Predicted vs Actual\n(Quantile Loss, Weighted)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_pred - self.y_test
        axes[0, 1].scatter(self.y_test, residuals, alpha=0.3, s=10, c='red')
        axes[0, 1].axhline(y=0, color='black', linestyle='--')
        axes[0, 1].set_xlabel('Actual Alpha')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].set_title('Residuals\n(Should be biased up with quantile=0.70)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution
        axes[0, 2].hist(self.y_test, bins=50, alpha=0.6, label='Actual', color='green')
        axes[0, 2].hist(y_pred, bins=50, alpha=0.6, label='Predicted', color='blue')
        axes[0, 2].set_xlabel('Alpha')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Alpha Distribution', fontweight='bold')
        axes[0, 2].legend()
        
        # Feature Importance
        top_features = self.feature_importances.head(12)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'], color='steelblue')
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'], fontsize=9)
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 12 Features', fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # Under-prediction analysis
        is_under = y_pred < self.y_test
        axes[1, 1].hist(self.y_test[is_under], bins=30, alpha=0.7, label='Under-predicted', color='red')
        axes[1, 1].hist(self.y_test[~is_under], bins=30, alpha=0.7, label='Over-predicted', color='green')
        axes[1, 1].set_xlabel('Actual Alpha')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Under vs Over-Prediction\n({self.metrics["under_pred_rate"]:.1f}% under-predicted)', fontweight='bold')
        axes[1, 1].legend()
        
        # Prediction by alpha region
        regions = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        mae_by_region = []
        region_labels = []
        
        for low, high in regions:
            mask = (self.y_test >= low) & (self.y_test < high)
            if np.sum(mask) > 0:
                mae = mean_absolute_error(self.y_test[mask], y_pred[mask])
                mae_by_region.append(mae)
                region_labels.append(f"[{low:.1f},{high:.1f})")
        
        axes[1, 2].bar(range(len(mae_by_region)), mae_by_region, color='purple', alpha=0.7)
        axes[1, 2].set_xticks(range(len(region_labels)))
        axes[1, 2].set_xticklabels(region_labels)
        axes[1, 2].set_ylabel('MAE')
        axes[1, 2].set_xlabel('Alpha Region')
        axes[1, 2].set_title('Error by Region', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_model(self, save_path='models/ml_hedge.pkl'):
        """Save model."""
        
        model_package = {
            'model': self.model,
            'model_type': 'gradient_boosting_quantile_weighted',
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'training_params': self.dataset['params']
        }
        
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Saved: {save_path}")
        
        return save_path


def main():
    """Training pipeline."""
    
    trainer = TwoStageTrainer(data_path='data/training_data.pkl')
    trainer.prepare_features()
    trainer.train_model()
    metrics = trainer.evaluate_model()
    trainer.visualize_results()
    trainer.save_model()
    
    print("\n" + "=" * 70)
    print("TWO-STAGE MODEL COMPLETE!")
    print("=" * 70)
    
    print(f"\nSummary:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Under-pred rate: {metrics['under_pred_rate']:.1f}%")
    

if __name__ == "__main__":
    main()