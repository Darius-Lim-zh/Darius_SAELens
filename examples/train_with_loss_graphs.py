#!/usr/bin/env python3
"""
Example script demonstrating SAE training with automatic loss graph generation.

This script shows how to train a Sparse Autoencoder using SAELens and automatically
generate loss function graphs at the end of training.
"""

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

def main():
    """Train an SAE with automatic loss graph generation."""
    
    # Check for available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Training parameters
    batch_size = 4096
    total_training_steps = 1000  # Reduced for demo
    total_training_tokens = total_training_steps * batch_size
    
    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5
    l1_warm_up_steps = total_training_steps // 20
    
    # Configuration
    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distribution)
        model_name="tiny-stories-1L-21M",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
        is_dataset_tokenized=True,
        streaming=True,
        
        # SAE Parameters
        sae=StandardTrainingSAEConfig(
            d_in=1024,
            d_sae=16 * 1024,  # expansion factor of 16
            apply_b_dec_to_input=True,
            normalize_activations="expected_average_only_in",
            l1_coefficient=5.0,
            lp_norm=1.0,
            l1_warm_up_steps=l1_warm_up_steps,
        ),
        
        # Training Parameters
        lr=5e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        train_batch_size_tokens=batch_size,
        context_size=256,
        
        # Activation Store Parameters
        n_batches_in_buffer=64,
        training_tokens=total_training_tokens,
        store_batch_size_prompts=16,
        
        # Resampling protocol args
        feature_sampling_window=100,
        dead_feature_window=100,
        dead_feature_threshold=1e-4,
        
        # WANDB Logging (optional)
        logger=LoggingConfig(
            log_to_wandb=False,  # Set to True if you want WandB logging
            wandb_project="sae_lens_demo",
            wandb_log_frequency=10,
            eval_every_n_wandb_logs=5,
        ),
        
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
    )
    
    print("Starting SAE training with automatic loss graph generation...")
    print(f"Model: {cfg.model_name}")
    print(f"Hook: {cfg.hook_name}")
    print(f"Training steps: {total_training_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Total tokens: {total_training_tokens:,}")
    
    # Run training
    runner = SAETrainingRunner(cfg)
    sparse_autoencoder = runner.run()
    
    print("\n🎉 Training completed!")
    print("Loss graphs have been automatically generated in the 'loss_graphs/' directory.")
    print("Check the console output above for details about the generated files.")
    
    return sparse_autoencoder

if __name__ == "__main__":
    main() 