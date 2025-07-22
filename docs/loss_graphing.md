# Loss Graphing Feature

The SAELens library now includes automatic loss graph generation during SAE training. This feature tracks all loss components during training and generates comprehensive visualizations at the end of training.

## Overview

The loss graphing feature automatically:

1. **Tracks Loss History**: Records all loss components during training (MSE loss, L1 loss, auxiliary losses, etc.)
2. **Generates Individual Graphs**: Creates separate plots for each loss type
3. **Creates Combined Graphs**: Produces multi-panel plots showing all losses together
4. **Saves Data**: Exports loss history to CSV files for further analysis
5. **Provides Statistics**: Calculates and displays summary statistics

## Generated Files

After training completes, the following files are automatically generated in the `loss_graphs/` directory:

### Individual Loss Graphs
- `loss_curves_mse_loss_{model}_{hook}_{timestamp}.png`
- `loss_curves_l1_loss_{model}_{hook}_{timestamp}.png`
- `loss_curves_overall_loss_{model}_{hook}_{timestamp}.png`
- `loss_curves_auxiliary_reconstruction_loss_{model}_{hook}_{timestamp}.png` (if applicable)

### Combined Graphs
- `loss_curves_combined_{model}_{hook}_{timestamp}.png` - Shows all losses on the same plot

### Data Files
- `loss_history_{model}_{hook}_{timestamp}.csv` - Raw loss data for further analysis

## Loss Types by SAE Architecture

### Standard SAE
- **MSE Loss**: Reconstruction loss (always present)
- **L1 Loss**: Sparsity penalty on weighted feature activations

### Gated SAE
- **MSE Loss**: Reconstruction loss
- **L1 Loss**: Sparsity penalty on gating activations
- **Auxiliary Reconstruction Loss**: Additional reconstruction path

### TopK SAE
- **MSE Loss**: Reconstruction loss
- **Auxiliary Reconstruction Loss**: Loss for dead neurons to learn useful features

## Usage

The loss graphing feature is automatically enabled when using `SAETrainingRunner`. No additional configuration is required:

```python
from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig, StandardTrainingSAEConfig

# Configure your training
cfg = LanguageModelSAERunnerConfig(
    # ... your configuration
    sae=StandardTrainingSAEConfig(
        # ... SAE configuration
    ),
)

# Run training - loss graphs will be generated automatically
runner = SAETrainingRunner(cfg)
sae = runner.run()

# Loss graphs are now available in the 'loss_graphs/' directory
```

## Example Output

After training, you'll see output like:

```
Generated 4 loss graph files:
  - loss_graphs/loss_curves_mse_loss_tiny-stories-1L-21M_blocks.0.hook_mlp_out_20241201_143022.png
  - loss_graphs/loss_curves_l1_loss_tiny-stories-1L-21M_blocks.0.hook_mlp_out_20241201_143022.png
  - loss_graphs/loss_curves_overall_loss_tiny-stories-1L-21M_blocks.0.hook_mlp_out_20241201_143022.png
  - loss_graphs/loss_curves_combined_tiny-stories-1L-21M_blocks.0.hook_mlp_out_20241201_143022.png
Saved loss history CSV to: loss_graphs/loss_history_tiny-stories-1L-21M_blocks.0.hook_mlp_out_20241201_143022.csv

Loss Training Summary:
  Model: tiny-stories-1L-21M
  Hook: blocks.0.hook_mlp_out
  Total Steps: 1000
  mse_loss:
    Final: 0.123456
    Mean: 0.234567
    Change: -0.111111 (-47.37%)
  l1_loss:
    Final: 0.045678
    Mean: 0.056789
    Change: 0.011111 (24.32%)
```

## Graph Features

### Individual Loss Graphs
- **Smoothing**: Moving average line for better trend visualization
- **Statistics**: Mean, final, min, max values displayed on graph
- **High Quality**: 300 DPI PNG files suitable for publications

### Combined Graphs
- **Linear Scale**: All losses on same scale for comparison
- **Log Scale**: Better visualization of different loss magnitudes
- **Legend**: Clear labeling of each loss type

### CSV Export
- **Raw Data**: Complete loss history for custom analysis
- **Compatible**: Standard CSV format for Excel, pandas, etc.
- **Structured**: Columns for steps and each loss type

## Configuration

The loss graphing feature uses sensible defaults but can be customized:

### Output Directory
Loss graphs are saved to `loss_graphs/` by default. You can modify this in the `LossGraphGenerator` class.

### File Naming
Files are named using the pattern:
`{prefix}_{model}_{hook}_{timestamp}.{extension}`

Where:
- `model` and `hook` are cleaned for filesystem compatibility
- `timestamp` is in format `YYYYMMDD_HHMMSS`

### Graph Styling
Graphs use seaborn styling with:
- Professional color schemes
- Clear typography
- Grid lines for readability
- High contrast for accessibility

## Advanced Usage

### Custom Loss Tracking
You can access the loss history directly from the trainer:

```python
# Get loss history
loss_history = trainer.get_loss_history()

# Generate custom graphs
from sae_lens.training.loss_graphing import LossGraphGenerator
graph_generator = LossGraphGenerator(output_dir="custom_graphs")
saved_files = graph_generator.generate_loss_graphs(loss_history)
```

### Summary Statistics
Get detailed statistics about the training:

```python
stats = graph_generator.generate_summary_stats(loss_history)
print(f"Final MSE loss: {stats['losses']['mse_loss']['final']}")
print(f"L1 loss change: {stats['losses']['l1_loss']['percent_change']:.2f}%")
```

## Troubleshooting

### No Loss Graphs Generated
- Check that training completed successfully
- Verify the `loss_graphs/` directory exists and is writable
- Look for warning messages in the console output

### Missing Loss Types
- Different SAE architectures have different loss types
- Check the architecture documentation for expected losses
- Verify the SAE configuration is correct

### Memory Issues
- Loss tracking uses minimal memory
- For very long training runs, consider disabling tracking: `trainer.disable_loss_tracking()`

## Dependencies

The loss graphing feature requires:
- `matplotlib` for plotting
- `seaborn` for styling
- `numpy` for data processing
- `pandas` for CSV export

These are typically included with standard scientific Python installations. 