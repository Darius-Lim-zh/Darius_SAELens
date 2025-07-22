"""
Tests for the loss graphing functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sae_lens.training.loss_graphing import LossGraphGenerator


class TestLossGraphGenerator:
    """Test the LossGraphGenerator class."""

    def test_init(self):
        """Test LossGraphGenerator initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LossGraphGenerator(output_dir=temp_dir)
            assert generator.output_dir == Path(temp_dir)
            assert generator.output_dir.exists()

    def test_generate_loss_graphs_empty_history(self):
        """Test handling of empty loss history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LossGraphGenerator(output_dir=temp_dir)
            result = generator.generate_loss_graphs({})
            assert result == []

    def test_generate_loss_graphs_missing_steps(self):
        """Test handling of loss history without steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LossGraphGenerator(output_dir=temp_dir)
            loss_history = {"mse_loss": [1.0, 2.0, 3.0]}
            result = generator.generate_loss_graphs(loss_history)
            assert result == []

    def test_generate_loss_graphs_basic(self):
        """Test basic loss graph generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LossGraphGenerator(output_dir=temp_dir)
            
            # Create sample loss history
            loss_history = {
                "steps": [0.0, 1.0, 2.0, 3.0, 4.0],
                "mse_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
                "l1_loss": [0.1, 0.2, 0.3, 0.35, 0.4],
                "overall_loss": [1.1, 1.0, 0.9, 0.85, 0.8],
            }
            
            result = generator.generate_loss_graphs(
                loss_history=loss_history,
                model_name="test_model",
                hook_name="test_hook",
                timestamp="20241201_120000"
            )
            
            # Should generate 4 files: 3 individual + 1 combined
            assert len(result) == 4
            
            # Check that files exist
            for file_path in result:
                assert Path(file_path).exists()
                assert Path(file_path).suffix == ".png"

    def test_generate_summary_stats(self):
        """Test summary statistics generation."""
        generator = LossGraphGenerator()
        
        loss_history = {
            "steps": [0.0, 1.0, 2.0, 3.0, 4.0],
            "mse_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
            "l1_loss": [0.1, 0.2, 0.3, 0.35, 0.4],
        }
        
        stats = generator.generate_summary_stats(
            loss_history=loss_history,
            model_name="test_model",
            hook_name="test_hook"
        )
        
        assert stats["model_name"] == "test_model"
        assert stats["hook_name"] == "test_hook"
        assert stats["total_steps"] == 5
        
        # Check MSE loss stats
        mse_stats = stats["losses"]["mse_loss"]
        assert mse_stats["mean"] == pytest.approx(0.66, rel=1e-2)
        assert mse_stats["final"] == 0.4
        assert mse_stats["initial"] == 1.0
        assert mse_stats["change"] == -0.6
        assert mse_stats["percent_change"] == -60.0
        
        # Check L1 loss stats
        l1_stats = stats["losses"]["l1_loss"]
        assert l1_stats["mean"] == pytest.approx(0.27, rel=1e-2)
        assert l1_stats["final"] == 0.4
        assert l1_stats["initial"] == 0.1
        assert l1_stats["change"] == 0.3
        assert l1_stats["percent_change"] == 300.0

    def test_save_loss_history_csv(self):
        """Test CSV export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LossGraphGenerator(output_dir=temp_dir)
            
            loss_history = {
                "steps": [0.0, 1.0, 2.0],
                "mse_loss": [1.0, 0.8, 0.6],
                "l1_loss": [0.1, 0.2, 0.3],
            }
            
            csv_file = generator.save_loss_history_csv(
                loss_history=loss_history,
                model_name="test_model",
                hook_name="test_hook",
                timestamp="20241201_120000"
            )
            
            assert Path(csv_file).exists()
            assert Path(csv_file).suffix == ".csv"
            
            # Check CSV content
            import pandas as pd
            df = pd.read_csv(csv_file)
            assert len(df) == 3
            assert list(df.columns) == ["steps", "mse_loss", "l1_loss"]

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_single_loss(self, mock_close, mock_savefig):
        """Test single loss plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LossGraphGenerator(output_dir=temp_dir)
            
            steps = [0.0, 1.0, 2.0, 3.0, 4.0]
            loss_values = [1.0, 0.8, 0.6, 0.5, 0.4]
            
            filepath = Path(temp_dir) / "test_loss.png"
            
            generator._plot_single_loss(
                steps=steps,
                loss_values=loss_values,
                loss_name="mse_loss",
                filepath=filepath,
                model_name="test_model",
                hook_name="test_hook"
            )
            
            # Verify plot was saved
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_combined_losses(self, mock_close, mock_savefig):
        """Test combined loss plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LossGraphGenerator(output_dir=temp_dir)
            
            loss_history = {
                "steps": [0, 1, 2, 3, 4],
                "mse_loss": [1.0, 0.8, 0.6, 0.5, 0.4],
                "l1_loss": [0.1, 0.2, 0.3, 0.35, 0.4],
            }
            
            filepath = Path(temp_dir) / "test_combined.png"
            
            generator._plot_combined_losses(
                loss_history=loss_history,
                filepath=filepath,
                model_name="test_model",
                hook_name="test_hook"
            )
            
            # Verify plot was saved
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    def test_file_naming(self):
        """Test file naming with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = LossGraphGenerator(output_dir=temp_dir)
            
            loss_history = {
                "steps": [0, 1],
                "mse_loss": [1.0, 0.8],
            }
            
            # Test with model/hook names that need cleaning
            result = generator.generate_loss_graphs(
                loss_history=loss_history,
                model_name="google/gemma-2b",
                hook_name="blocks.0.hook_mlp_out",
                timestamp="20241201_120000"
            )
            
            # Check that files were created with cleaned names
            assert len(result) == 2  # individual + combined
            
            for file_path in result:
                assert "google__gemma-2b" in file_path
                assert "blocks.0.hook_mlp_out" in file_path
                assert "20241201_120000" in file_path

    def test_loss_tracking_integration(self):
        """Test that loss tracking integrates with the trainer."""
        # This test verifies that the loss tracking methods exist on the trainer
        from sae_lens.training.sae_trainer import SAETrainer
        
        # Create a mock trainer
        mock_trainer = Mock(spec=SAETrainer)
        mock_trainer.get_loss_history.return_value = {
            "steps": [0, 1, 2],
            "mse_loss": [1.0, 0.8, 0.6],
            "l1_loss": [0.1, 0.2, 0.3],
        }
        
        # Test that we can get loss history
        loss_history = mock_trainer.get_loss_history()
        assert "steps" in loss_history
        assert "mse_loss" in loss_history
        assert "l1_loss" in loss_history 