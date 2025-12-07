#!/usr/bin/env python3
"""
Encode a single language instruction into a T5-XXL embedding and save it to disk.

Run from the RoboticsDiffusionTransformer repo root as:
    conda activate rdt
    python scripts/encode_lang.py
"""

import os
import sys
import torch
import yaml

# Add repo root to sys.path so `models` can be imported when run as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.multimodal_encoder.t5_encoder import T5Embedder

# GPU index to use
GPU = 0

# Path to the T5-XXL weights.
# If you downloaded them under repo root as google/t5-v1_1-xxl, use this:
MODEL_PATH = "google/t5-v1_1-xxl"
# If instead you put them somewhere else, e.g. ../../encoders/t5-v1_1-xxl, change to that.

# RDT base config (used to get tokenizer_max_length)
CONFIG_PATH = "configs/base.yaml"

# Where to save embeddings
SAVE_DIR = "outs/lang_embeddings"

# If VRAM < 24 GB, offload large parts of T5-XXL to CPU/disk
OFFLOAD_DIR = "outs/t5_offload"  # make sure this directory exists


def encode_lang(instruction: str, task_name: str) -> str:
    """Encode a single instruction and save embeddings to SAVE_DIR/task_name.pt."""
    # Load config (for tokenizer_max_length)
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)

    # Ensure output dirs exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OFFLOAD_DIR, exist_ok=True)

    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    # Create T5 embedder (with offloading)
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=OFFLOAD_DIR,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    # Tokenize instruction
    tokens = tokenizer(
        instruction,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"].to(device)

    tokens = tokens.view(1, -1)

    # Forward pass
    with torch.no_grad():
        pred = text_encoder(tokens).last_hidden_state.detach().cpu()

    # Save to disk in the format RDT expects
    save_path = os.path.join(SAVE_DIR, f"{task_name}.pt")
    torch.save(
        {
            "name": task_name,
            "instruction": instruction,
            "embeddings": pred,
        },
        save_path,
    )

    print(
        f'"{instruction}" from "{task_name}" is encoded by "{MODEL_PATH}" '
        f'into shape {pred.shape} and saved to "{save_path}"'
    )
    return save_path


def main():
    # Define the task name + instruction used for fine-tuning
    task_name = "berkeley_ur5"
    instruction = "Pick and place objects using the UR5 arm."

    encode_lang(instruction, task_name)


if __name__ == "__main__":
    main()
