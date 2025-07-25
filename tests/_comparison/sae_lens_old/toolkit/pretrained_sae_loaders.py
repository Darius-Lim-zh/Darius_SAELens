import json
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import load_file

from tests._comparison.sae_lens_old import logger
from tests._comparison.sae_lens_old.config import (
    DTYPE_MAP,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
)
from tests._comparison.sae_lens_old.toolkit.pretrained_saes_directory import (
    get_config_overrides,
    get_pretrained_saes_directory,
    get_repo_id_and_folder_name,
)


# loaders take in a release, sae_id, device, and whether to force download, and returns a tuple of config, state_dict, and log sparsity
class PretrainedSaeHuggingfaceLoader(Protocol):
    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str,
        force_download: bool,
        cfg_overrides: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]: ...


class PretrainedSaeConfigHuggingfaceLoader(Protocol):
    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str,
        force_download: bool,
        cfg_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


class PretrainedSaeDiskLoader(Protocol):
    def __call__(
        self,
        path: str | Path,
        device: str,
        cfg_overrides: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]: ...


class PretrainedSaeConfigDiskLoader(Protocol):
    def __call__(
        self,
        path: str | Path,
        device: str | None,
        cfg_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


def sae_lens_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Loads SAEs from Hugging Face"""
    cfg_dict = get_sae_lens_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    weights_filename = f"{folder_name}/{SAE_WEIGHTS_FILENAME}"
    sae_path = hf_hub_download(
        repo_id=repo_id, filename=weights_filename, force_download=force_download
    )

    try:
        sparsity_filename = f"{folder_name}/{SPARSITY_FILENAME}"
        log_sparsity_path = hf_hub_download(
            repo_id=repo_id, filename=sparsity_filename, force_download=force_download
        )
    except EntryNotFoundError:
        log_sparsity_path = None  # no sparsity file

    cfg_dict, state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=sae_path,
        device=device,
    )

    # get sparsity tensor if it exists
    if log_sparsity_path is not None:
        with safe_open(log_sparsity_path, framework="pt", device=device) as f:  # type: ignore
            log_sparsity = f.get_tensor("sparsity")
    else:
        log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


def sae_lens_disk_loader(
    path: str | Path,
    device: str = "cpu",
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Loads SAEs from disk"""

    weights_path = Path(path) / SAE_WEIGHTS_FILENAME
    cfg_dict = get_sae_lens_config_from_disk(path, device, cfg_overrides)
    cfg_dict, state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weights_path,
        device=device,
    )
    return cfg_dict, state_dict


def get_sae_lens_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Retrieve the configuration for a Sparse Autoencoder (SAE) from a Hugging Face repository.

    Args:
        repo_id (str): The repository ID on Hugging Face.
        folder_name (str): The folder name within the repository containing the config file.
        force_download (bool, optional): Whether to force download the config file. Defaults to False.
        cfg_overrides (dict[str, Any] | None, optional): Overrides for the config. Defaults to None.

    Returns:
        dict[str, Any]: The configuration dictionary for the SAE.
    """
    cfg_filename = f"{folder_name}/{SAE_CFG_FILENAME}"
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=cfg_filename, force_download=force_download
    )
    sae_path = Path(cfg_path).parent
    return get_sae_lens_config_from_disk(sae_path, device, cfg_overrides)


def get_sae_lens_config_from_disk(
    path: str | Path,
    device: str | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg_filename = Path(path) / SAE_CFG_FILENAME

    with open(cfg_filename) as f:
        cfg_dict: dict[str, Any] = json.load(f)

    if device is not None:
        cfg_dict["device"] = device

    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)

    return cfg_dict


def handle_config_defaulting(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    # Set default values for backwards compatibility
    cfg_dict.setdefault("prepend_bos", True)
    cfg_dict.setdefault("dataset_trust_remote_code", True)
    cfg_dict.setdefault("apply_b_dec_to_input", True)
    cfg_dict.setdefault("finetuning_scaling_factor", False)
    cfg_dict.setdefault("sae_lens_training_version", None)
    cfg_dict.setdefault("activation_fn_str", cfg_dict.get("activation_fn", "relu"))
    cfg_dict.setdefault("architecture", "standard")
    cfg_dict.setdefault("neuronpedia_id", None)

    if "normalize_activations" in cfg_dict and isinstance(
        cfg_dict["normalize_activations"], bool
    ):
        # backwards compatibility
        cfg_dict["normalize_activations"] = (
            "none"
            if not cfg_dict["normalize_activations"]
            else "expected_average_only_in"
        )

    cfg_dict.setdefault("normalize_activations", "none")
    cfg_dict.setdefault("device", "cpu")

    return cfg_dict


def get_connor_rob_hook_z_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = folder_name.split(".pt")[0] + "_cfg.json"
    config_path = hf_hub_download(repo_id, config_path, force_download=force_download)

    with open(config_path) as config_file:
        old_cfg_dict = json.load(config_file)

    return {
        "architecture": "standard",
        "d_in": old_cfg_dict["act_size"],
        "d_sae": old_cfg_dict["dict_size"],
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": "gpt2-small",
        "hook_name": old_cfg_dict["act_name"],
        "hook_layer": old_cfg_dict["layer"],
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "Skylion007/openwebtext",
        "context_size": 128,
        "normalize_activations": "none",
        "dataset_trust_remote_code": True,
        **(cfg_overrides or {}),
    }


def connor_rob_hook_z_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], None]:
    cfg_dict = get_connor_rob_hook_z_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    file_path = hf_hub_download(
        repo_id=repo_id, filename=folder_name, force_download=force_download
    )
    weights = torch.load(file_path, map_location=device)

    return cfg_dict, weights, None


def read_sae_components_from_disk(
    cfg_dict: dict[str, Any],
    weight_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Given a loaded dictionary and a path to a weight file, load the weights and return the state_dict.
    """
    if dtype is None:
        dtype = DTYPE_MAP[cfg_dict["dtype"]]

    state_dict = {}
    with safe_open(weight_path, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k).to(dtype=dtype)

    # if bool and True, then it's the April update method of normalizing activations and hasn't been folded in.
    if "scaling_factor" in state_dict:
        # we were adding it anyway for a period of time but are no longer doing so.
        # so we should delete it if
        if torch.allclose(
            state_dict["scaling_factor"],
            torch.ones_like(state_dict["scaling_factor"]),
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            if not cfg_dict["finetuning_scaling_factor"]:
                raise ValueError(
                    "Scaling factor is present but finetuning_scaling_factor is False."
                )
            state_dict["finetuning_scaling_factor"] = state_dict["scaling_factor"]
            del state_dict["scaling_factor"]
    else:
        # it's there and it's not all 1's, we should use it.
        cfg_dict["finetuning_scaling_factor"] = False

    return cfg_dict, state_dict


def get_gemma_2_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Detect width from folder_name
    width_map = {
        "width_4k": 4096,
        "width_16k": 16384,
        "width_32k": 32768,
        "width_65k": 65536,
        "width_131k": 131072,
        "width_262k": 262144,
        "width_524k": 524288,
        "width_1m": 1048576,
    }
    d_sae = next(
        (width for key, width in width_map.items() if key in folder_name), None
    )

    # Detect layer from folder_name
    match = re.search(r"layer_(\d+)", folder_name)
    layer = int(match.group(1)) if match else None
    if layer is None:
        if "embedding" in folder_name:
            layer = 0
        else:
            raise ValueError("Layer not found in folder_name and no override provided.")

    # Model specific parameters
    model_params = {
        "2b-it": {"name": "gemma-2-2b-it", "d_in": 2304},
        "9b-it": {"name": "gemma-2-9b-it", "d_in": 3584},
        "27b-it": {"name": "gemma-2-27b-it", "d_in": 4608},
        "2b": {"name": "gemma-2-2b", "d_in": 2304},
        "9b": {"name": "gemma-2-9b", "d_in": 3584},
        "27b": {"name": "gemma-2-27b", "d_in": 4608},
    }
    model_info = next(
        (info for key, info in model_params.items() if key in repo_id), None
    )
    if not model_info:
        raise ValueError("Model name not found in repo_id.")

    model_name, d_in = model_info["name"], model_info["d_in"]

    # Hook specific parameters
    if "res" in repo_id and "embedding" not in folder_name:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif "res" in repo_id and "embedding" in folder_name:
        hook_name = "hook_embed"
    elif "mlp" in repo_id:
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif "att" in repo_id:
        hook_name = f"blocks.{layer}.attn.hook_z"
        d_in = {"2b": 2048, "9b": 4096, "27b": 4608}.get(
            next(key for key in model_params if key in repo_id), d_in
        )
    else:
        raise ValueError("Hook name not found in folder_name.")

    cfg = {
        "architecture": "jumprelu",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": model_name,
        "hook_name": hook_name,
        "hook_layer": layer,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
    }
    if device is not None:
        cfg["device"] = device

    if cfg_overrides is not None:
        cfg.update(cfg_overrides)

    return cfg


def gemma_2_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Custom loader for Gemma 2 SAEs.
    """
    cfg_dict = get_gemma_2_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename="params.npz",
        subfolder=folder_name,
        force_download=force_download,
    )

    # Load and convert the weights
    state_dict = {}
    with np.load(sae_path) as data:
        for key in data:
            state_dict_key = "W_" + key[2:] if key.startswith("w_") else key
            state_dict[state_dict_key] = (
                torch.tensor(data[key]).to(dtype=torch.float32).to(device)
            )

    # Handle scaling factor
    if "scaling_factor" in state_dict:
        if torch.allclose(
            state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            if not cfg_dict["finetuning_scaling_factor"]:
                raise ValueError(
                    "Scaling factor is present but finetuning_scaling_factor is False."
                )
            state_dict["finetuning_scaling_factor"] = state_dict.pop("scaling_factor")
    else:
        cfg_dict["finetuning_scaling_factor"] = False

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    # if it is an embedding SAE, then we need to adjust for the scale of d_model because of how they trained it
    if "embedding" in folder_name:
        logger.debug("Adjusting for d_model in embedding SAE")
        state_dict["W_enc"].data = state_dict["W_enc"].data / np.sqrt(cfg_dict["d_in"])
        state_dict["W_dec"].data = state_dict["W_dec"].data * np.sqrt(cfg_dict["d_in"])

    return cfg_dict, state_dict, log_sparsity


def get_llama_scope_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Llama Scope SAEs
    # repo_id: fnlp/Llama3_1-8B-Base-LX{sublayer}-{exp_factor}x
    # folder_name: Llama3_1-8B-Base-L{layer}{sublayer}-{exp_factor}x
    config_path = folder_name + "/hyperparams.json"
    config_path = hf_hub_download(repo_id, config_path, force_download=force_download)

    with open(config_path) as f:
        old_cfg_dict = json.load(f)

    # Model specific parameters
    model_name, d_in = "meta-llama/Llama-3.1-8B", old_cfg_dict["d_model"]

    cfg_dict = {
        "architecture": "jumprelu",
        "jump_relu_threshold": old_cfg_dict["jump_relu_threshold"],
        # We use a scalar jump_relu_threshold for all features
        # This is different from Gemma Scope JumpReLU SAEs.
        "d_in": d_in,
        "d_sae": old_cfg_dict["d_sae"],
        "dtype": "bfloat16",
        "model_name": model_name,
        "hook_name": old_cfg_dict["hook_point_in"],
        "hook_layer": int(old_cfg_dict["hook_point_in"].split(".")[1]),
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "cerebras/SlimPajama-627B",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": "expected_average_only_in",
    }

    if device is not None:
        cfg_dict["device"] = device

    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)

    return cfg_dict


def llama_scope_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Custom loader for Llama Scope SAEs.

    Args:
        release: Release identifier
        sae_id: SAE identifier
        device: Device to load tensors to
        force_download: Whether to force download even if files exist
        cfg_overrides: Configuration overrides (optional)
        d_sae_override: Override for SAE dimension (optional)
        layer_override: Override for layer number (optional)

    Returns:
        tuple of (config dict, state dict, log sparsity tensor)
    """
    cfg_dict = get_llama_scope_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename="final.safetensors",
        subfolder=folder_name + "/checkpoints",
        force_download=force_download,
    )

    # Load the weights using load_file instead of safe_open
    state_dict_loaded = load_file(sae_path, device=device)

    # Convert and organize the weights
    state_dict = {
        "W_enc": state_dict_loaded["encoder.weight"]
        .to(dtype=DTYPE_MAP[cfg_dict["dtype"]])
        .T,
        "W_dec": state_dict_loaded["decoder.weight"]
        .to(dtype=DTYPE_MAP[cfg_dict["dtype"]])
        .T,
        "b_enc": state_dict_loaded["encoder.bias"].to(
            dtype=DTYPE_MAP[cfg_dict["dtype"]]
        ),
        "b_dec": state_dict_loaded["decoder.bias"].to(
            dtype=DTYPE_MAP[cfg_dict["dtype"]]
        ),
        "threshold": torch.ones(
            cfg_dict["d_sae"],
            dtype=DTYPE_MAP[cfg_dict["dtype"]],
            device=cfg_dict["device"],
        )
        * cfg_dict["jump_relu_threshold"],
    }

    # No sparsity tensor for Llama Scope SAEs
    log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


def get_dictionary_learning_config_1_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Suitable for SAEs from https://huggingface.co/canrager/lm_sae.
    """
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{folder_name}/config.json",
        force_download=force_download,
    )
    with open(config_path) as f:
        config = json.load(f)

    trainer = config["trainer"]
    buffer = config["buffer"]

    hook_point_name = f"blocks.{trainer['layer']}.hook_resid_post"

    activation_fn_str = "topk" if trainer["dict_class"] == "AutoEncoderTopK" else "relu"
    activation_fn_kwargs = {"k": trainer["k"]} if activation_fn_str == "topk" else {}

    return {
        "architecture": (
            "gated" if trainer["dict_class"] == "GatedAutoEncoder" else "standard"
        ),
        "d_in": trainer["activation_dim"],
        "d_sae": trainer["dict_size"],
        "dtype": "float32",
        "device": device,
        "model_name": trainer["lm_name"].split("/")[-1],
        "hook_name": hook_point_name,
        "hook_layer": trainer["layer"],
        "hook_head_index": None,
        "activation_fn_str": activation_fn_str,
        "activation_fn_kwargs": activation_fn_kwargs,
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": buffer["ctx_len"],
        "normalize_activations": "none",
        "neuronpedia_id": None,
        "dataset_trust_remote_code": True,
        **(cfg_overrides or {}),
    }


def get_deepseek_r1_config_from_hf(
    repo_id: str,  # noqa: ARG001
    folder_name: str,
    device: str,
    force_download: bool = False,  # noqa: ARG001
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get config for DeepSeek R1 SAEs."""

    match = re.search(r"l(\d+)", folder_name)
    if match is None:
        raise ValueError(f"Could not find layer number in filename: {folder_name}")
    layer = int(match.group(1))

    return {
        "architecture": "standard",
        "d_in": 4096,  # LLaMA 8B hidden size
        "d_sae": 4096 * 16,  # Expansion factor 16
        "dtype": "bfloat16",
        "context_size": 1024,
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "hook_layer": layer,
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": None,
        "activation_fn_str": "relu",
        "normalize_activations": "none",
        "device": device,
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,
        **(cfg_overrides or {}),
    }


def deepseek_r1_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Load a DeepSeek R1 SAE."""
    # Download weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=folder_name,
        force_download=force_download,
    )

    # Load state dict
    state_dict_loaded = torch.load(sae_path, map_location=device)

    # Create config
    cfg_dict = get_deepseek_r1_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Convert weights
    state_dict = {
        "W_enc": state_dict_loaded["encoder.weight"].T,
        "W_dec": state_dict_loaded["decoder.weight"].T,
        "b_enc": state_dict_loaded["encoder.bias"],
        "b_dec": state_dict_loaded["decoder.bias"],
    }

    return cfg_dict, state_dict, None


def get_conversion_loader_name(release: str) -> str:
    saes_directory = get_pretrained_saes_directory()
    sae_info = saes_directory.get(release, None)
    conversion_loader_name = "sae_lens"
    if sae_info is not None and sae_info.conversion_func is not None:
        conversion_loader_name = sae_info.conversion_func
    if conversion_loader_name not in NAMED_PRETRAINED_SAE_LOADERS:
        raise ValueError(
            f"Conversion func '{conversion_loader_name}' not found in NAMED_PRETRAINED_SAE_LOADERS."
        )
    return conversion_loader_name


def load_sae_config_from_huggingface(
    release: str,
    sae_id: str,
    device: str = "cpu",
    force_download: bool = False,
) -> dict[str, Any]:
    cfg_overrides = get_config_overrides(release, sae_id)
    conversion_loader_name = get_conversion_loader_name(release)
    config_getter = NAMED_PRETRAINED_SAE_CONFIG_GETTERS[conversion_loader_name]
    repo_id, folder_name = get_repo_id_and_folder_name(release, sae_id=sae_id)
    cfg = {
        **config_getter(repo_id, folder_name, device, force_download, cfg_overrides),
    }
    return handle_config_defaulting(cfg)


def dictionary_learning_sae_huggingface_loader_1(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Suitable for SAEs from https://huggingface.co/canrager/lm_sae.
    """
    cfg_dict = get_dictionary_learning_config_1_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    encoder_path = hf_hub_download(
        repo_id=repo_id, filename=f"{folder_name}/ae.pt", force_download=force_download
    )
    encoder = torch.load(encoder_path, map_location="cpu")

    state_dict = {
        "W_enc": encoder["encoder.weight"].T,
        "W_dec": encoder["decoder.weight"].T,
        "b_dec": encoder.get(
            "b_dec", encoder.get("bias", encoder.get("decoder_bias", None))
        ),
    }

    if "encoder.bias" in encoder:
        state_dict["b_enc"] = encoder["encoder.bias"]

    if "mag_bias" in encoder:
        state_dict["b_mag"] = encoder["mag_bias"]
    if "gate_bias" in encoder:
        state_dict["b_gate"] = encoder["gate_bias"]
    if "r_mag" in encoder:
        state_dict["r_mag"] = encoder["r_mag"]

    return cfg_dict, state_dict, None


def get_llama_scope_r1_distill_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Future Llama Scope series SAE by OpenMoss group use this config.
    # repo_id: [
    #   fnlp/Llama-Scope-R1-Distill
    # ]
    # folder_name: [
    #   800M-Slimpajama-0-OpenR1-Math-220k/L{layer}R,
    #   400M-Slimpajama-400M-OpenR1-Math-220k/L{layer}R,
    #   0-Slimpajama-800M-OpenR1-Math-220k/L{layer}R,
    # ]
    config_path = folder_name + "/config.json"
    config_path = hf_hub_download(repo_id, config_path, force_download=force_download)

    with open(config_path) as f:
        huggingface_cfg_dict = json.load(f)

    # Model specific parameters
    model_name, d_in = "meta-llama/Llama-3.1-8B", huggingface_cfg_dict["d_model"]

    return {
        "architecture": "jumprelu",
        "d_in": d_in,
        "d_sae": d_in * huggingface_cfg_dict["expansion_factor"],
        "dtype": "float32",
        "device": device,
        "model_name": model_name,
        "hook_name": huggingface_cfg_dict["hook_point_in"],
        "hook_layer": int(huggingface_cfg_dict["hook_point_in"].split(".")[1]),
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "cerebras/SlimPajama-627B",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": "expected_average_only_in",
        **(cfg_overrides or {}),
    }


def llama_scope_r1_distill_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Custom loader for Llama Scope SAEs.

    Args:
        release: Release identifier
        sae_id: SAE identifier
        device: Device to load tensors to
        force_download: Whether to force download even if files exist
        cfg_overrides: Configuration overrides (optional)
        d_sae_override: Override for SAE dimension (optional)
        layer_override: Override for layer number (optional)

    Returns:
        tuple of (config dict, state dict, log sparsity tensor)
    """
    cfg_dict = get_llama_scope_r1_distill_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename=SAE_WEIGHTS_FILENAME,
        subfolder=folder_name,
        force_download=force_download,
    )

    # Load the weights using load_file instead of safe_open
    state_dict_loaded = load_file(sae_path, device=device)

    # Convert and organize the weights
    state_dict = {
        "W_enc": state_dict_loaded["encoder.weight"]
        .to(dtype=DTYPE_MAP[cfg_dict["dtype"]])
        .T,
        "W_dec": state_dict_loaded["decoder.weight"]
        .to(dtype=DTYPE_MAP[cfg_dict["dtype"]])
        .T,
        "b_enc": state_dict_loaded["encoder.bias"].to(
            dtype=DTYPE_MAP[cfg_dict["dtype"]]
        ),
        "b_dec": state_dict_loaded["decoder.bias"].to(
            dtype=DTYPE_MAP[cfg_dict["dtype"]]
        ),
        "threshold": state_dict_loaded["log_jumprelu_threshold"]
        .to(dtype=DTYPE_MAP[cfg_dict["dtype"]])
        .exp(),
    }

    # No sparsity tensor for Llama Scope SAEs
    log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


NAMED_PRETRAINED_SAE_LOADERS: dict[str, PretrainedSaeHuggingfaceLoader] = {
    "sae_lens": sae_lens_huggingface_loader,
    "connor_rob_hook_z": connor_rob_hook_z_huggingface_loader,
    "gemma_2": gemma_2_sae_huggingface_loader,
    "llama_scope": llama_scope_sae_huggingface_loader,
    "llama_scope_r1_distill": llama_scope_r1_distill_sae_huggingface_loader,
    "dictionary_learning_1": dictionary_learning_sae_huggingface_loader_1,
    "deepseek_r1": deepseek_r1_sae_huggingface_loader,
}


NAMED_PRETRAINED_SAE_CONFIG_GETTERS: dict[str, PretrainedSaeConfigHuggingfaceLoader] = {
    "sae_lens": get_sae_lens_config_from_hf,
    "connor_rob_hook_z": get_connor_rob_hook_z_config_from_hf,
    "gemma_2": get_gemma_2_config_from_hf,
    "llama_scope": get_llama_scope_config_from_hf,
    "llama_scope_r1_distill": get_llama_scope_r1_distill_config_from_hf,
    "dictionary_learning_1": get_dictionary_learning_config_1_from_hf,
    "deepseek_r1": get_deepseek_r1_config_from_hf,
}
