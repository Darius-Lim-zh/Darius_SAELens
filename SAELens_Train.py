import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from sae_lens import SAETrainingRunner, LanguageModelSAERunnerConfig
from sae_lens.saes import StandardTrainingSAEConfig
import torch

# Tutorial default configuration
model_name = "tiny-stories-1L-21M"
dataset_path = "apollo-research/roneneldan-TinyStories-tokenizer-gpt2"

# Training parameters
batch_size = 4096
total_training_steps = 30000
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5
l1_warm_up_steps = total_training_steps // 20

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distribution)
    model_name=model_name,
    hook_name="blocks.0.hook_mlp_out",
    dataset_path=dataset_path,
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
    feature_sampling_window=1000,
    dead_feature_window=1000,
    dead_feature_threshold=1e-4,
    
    # WANDB Logging
    # log_to_wandb=False,  # Temporarily disabled for testing
    # wandb_project="sae_lens_tutorial",
    # wandb_log_frequency=30,
    # eval_every_n_wandb_logs=20,
    
    # Misc
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)

runner = SAETrainingRunner(cfg)
sparse_autoencoder = runner.run()

# Upload to Huggingface with error handling and backup
from sae_lens import upload_saes_to_huggingface
import torch
import datetime

# If you have multiple SAEs, add them to this dictionary
saes_dict = {
    cfg.hook_name: sparse_autoencoder
}

upload_saes_to_huggingface(
    saes_dict,
    hf_repo_id="dariuslimzh/test_SAE"
)

print("\n🎉 Training completed! Your SAE is safely backed up locally.") 