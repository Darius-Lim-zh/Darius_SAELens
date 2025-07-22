"""
Loss graphing functionality for SAE training.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class LossGraphGenerator:
    """Generate and save loss graphs from training history."""

    def __init__(self, output_dir: str = "loss_graphs"):
        """
        Initialize the loss graph generator.
        
        Args:
            output_dir: Directory to save the loss graphs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def generate_loss_graphs(
        self,
        loss_history: Dict[str, List[float]],
        model_name: str = "unknown",
        hook_name: str = "unknown",
        timestamp: str | None = None,
    ) -> List[str]:
        """
        Generate and save loss graphs from training history.
        Returns a list of saved file paths. Skips any loss arrays that do not match the length of steps.
        """
        if not loss_history or "steps" not in loss_history:
            print("No loss history provided or missing steps data")
            return []
            
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
        model_name_clean = model_name.replace("/", "_").replace(" ", "_")
        hook_name_clean = hook_name.replace("/", "_").replace(" ", "_")
        
        saved_files = []
        skipped_losses = []
        steps_len = len(loss_history["steps"])
        
        # Generate individual loss graphs
        for loss_name, loss_values in loss_history.items():
            if loss_name == "steps" or not loss_values:
                continue
            if len(loss_values) != steps_len:
                print(f"Skipping {loss_name} (length {len(loss_values)}) due to length mismatch with steps ({steps_len})")
                skipped_losses.append(loss_name)
                continue
            filename = f"loss_curves_{loss_name}_{model_name_clean}_{hook_name_clean}_{timestamp}.png"
            filepath = self.output_dir / filename
            self._plot_single_loss(
                steps=loss_history["steps"],
                loss_values=loss_values,
                loss_name=loss_name,
                filepath=filepath,
                model_name=model_name,
                hook_name=hook_name,
            )
            saved_files.append(str(filepath))
        # Generate combined loss graph (only for valid losses)
        combined_filename = f"loss_curves_combined_{model_name_clean}_{hook_name_clean}_{timestamp}.png"
        combined_filepath = self.output_dir / combined_filename
        valid_loss_history = {k: v for k, v in loss_history.items() if k == "steps" or (k not in skipped_losses and len(v) == steps_len)}
        self._plot_combined_losses(
            loss_history=valid_loss_history,
            filepath=combined_filepath,
            model_name=model_name,
            hook_name=hook_name,
        )
        saved_files.append(str(combined_filepath))
        if skipped_losses:
            print(f"Skipped plotting for losses due to length mismatch: {', '.join(skipped_losses)}")
        return saved_files

    def _plot_single_loss(
        self,
        steps: List[float],
        loss_values: List[float],
        loss_name: str,
        filepath: Path,
        model_name: str,
        hook_name: str,
    ) -> None:
        """Plot a single loss curve."""
        plt.figure(figsize=(12, 8))
        
        # Convert to numpy arrays for easier manipulation
        steps_array = np.array(steps)
        loss_array = np.array(loss_values)
        
        # Create the plot
        plt.plot(steps_array, loss_array, linewidth=2, alpha=0.8)
        
        # Add smoothing line if enough data points
        if len(steps_array) > 10:
            # Calculate moving average
            window_size = max(1, len(steps_array) // 20)
            smoothed = np.convolve(loss_array, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = steps_array[window_size-1:]
            plt.plot(smoothed_steps, smoothed, linewidth=3, alpha=0.6, linestyle='--', 
                    label=f'Moving Average (window={window_size})')
        
        # Customize the plot
        plt.title(f'{loss_name.replace("_", " ").title()} Loss Over Training Steps\n'
                 f'Model: {model_name} | Hook: {hook_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel(f'{loss_name.replace("_", " ").title()} Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics
        mean_loss = np.mean(loss_array)
        final_loss = loss_array[-1]
        min_loss = np.min(loss_array)
        max_loss = np.max(loss_array)
        
        stats_text = f'Mean: {mean_loss:.6f}\nFinal: {final_loss:.6f}\nMin: {min_loss:.6f}\nMax: {max_loss:.6f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {loss_name} loss graph to: {filepath}")

    def _plot_combined_losses(
        self,
        loss_history: Dict[str, List[float]],
        filepath: Path,
        model_name: str,
        hook_name: str,
    ) -> None:
        """Plot all losses on the same graph."""
        plt.figure(figsize=(14, 10))
        
        # Get all loss types except steps
        loss_types = [k for k in loss_history.keys() if k != "steps"]
        
        if not loss_types:
            print("No loss types found for combined plot")
            return
            
        # Create subplots
        _, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Main plot with all losses
        steps_array = np.array(loss_history["steps"])
        
        for loss_name in loss_types:
            if loss_name in loss_history and loss_history[loss_name]:
                loss_array = np.array(loss_history[loss_name])
                if len(loss_array) == len(steps_array):
                    axes[0].plot(steps_array, loss_array, linewidth=2, alpha=0.8, 
                               label=loss_name.replace("_", " ").title())
        
        axes[0].set_title(f'All Losses Over Training Steps\n'
                         f'Model: {model_name} | Hook: {hook_name}', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Training Steps', fontsize=12)
        axes[0].set_ylabel('Loss Value', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Log scale plot for better visualization of different loss scales
        for loss_name in loss_types:
            if loss_name in loss_history and loss_history[loss_name]:
                loss_array = np.array(loss_history[loss_name])
                if len(loss_array) == len(steps_array):
                    # Only plot positive values for log scale
                    positive_mask = loss_array > 0
                    if np.any(positive_mask):
                        axes[1].plot(steps_array[positive_mask], loss_array[positive_mask], 
                                   linewidth=2, alpha=0.8, 
                                   label=loss_name.replace("_", " ").title())
        
        axes[1].set_title(f'All Losses (Log Scale) - Positive Values Only\n'
                         f'Model: {model_name} | Hook: {hook_name}', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Training Steps', fontsize=12)
        axes[1].set_ylabel('Loss Value (Log Scale)', fontsize=12)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined loss graph to: {filepath}")

    def generate_summary_stats(
        self,
        loss_history: Dict[str, List[float]],
        model_name: str = "unknown",
        hook_name: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for the loss history.
        
        Args:
            loss_history: Dictionary containing loss history
            model_name: Name of the model
            hook_name: Name of the hook
            
        Returns:
            Dictionary containing summary statistics
        """
        stats = {
            "model_name": model_name,
            "hook_name": hook_name,
            "total_steps": len(loss_history.get("steps", [])),
            "losses": {}
        }
        
        for loss_name, loss_values in loss_history.items():
            if loss_name == "steps" or not loss_values:
                continue
                
            loss_array = np.array(loss_values)
            stats["losses"][loss_name] = {
                "mean": float(np.mean(loss_array)),
                "std": float(np.std(loss_array)),
                "min": float(np.min(loss_array)),
                "max": float(np.max(loss_array)),
                "final": float(loss_array[-1]),
                "initial": float(loss_array[0]),
                "change": float(loss_array[-1] - loss_array[0]),
                "percent_change": float(((loss_array[-1] - loss_array[0]) / loss_array[0]) * 100) if loss_array[0] != 0 else 0.0
            }
            
        return stats

    def save_loss_history_csv(
        self,
        loss_history: Dict[str, List[float]],
        model_name: str = "unknown",
        hook_name: str = "unknown",
        timestamp: str | None = None,
    ) -> str:
        """
        Save loss history to CSV file.
        
        Args:
            loss_history: Dictionary containing loss history
            model_name: Name of the model
            hook_name: Name of the hook
            timestamp: Optional timestamp for file naming
            
        Returns:
            Path to saved CSV file
        """
        import pandas as pd
        
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Clean up names for file naming
        model_name_clean = model_name.replace("/", "_").replace(" ", "_")
        hook_name_clean = hook_name.replace("/", "_").replace(" ", "_")
        
        filename = f"loss_history_{model_name_clean}_{hook_name_clean}_{timestamp}.csv"
        filepath = self.output_dir / filename
        
        # Convert to DataFrame
        df = pd.DataFrame(loss_history)
        df.to_csv(filepath, index=False)
        
        print(f"Saved loss history CSV to: {filepath}")
        return str(filepath) 