#!/usr/bin/env bash

export CUDA_HOME=/usr
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt-170m-berkeley"
export WANDB_PROJECT="robotics_diffusion_transformer"

mkdir -p "$OUTPUT_DIR"

deepspeed main.py \
  --deepspeed="./configs/zero2.json" \
  --pretrained_model_name_or_path="robotics-diffusion-transformer/rdt-170m" \
  --pretrained_text_encoder_name_or_path="$TEXT_ENCODER_NAME" \
  --pretrained_vision_encoder_name_or_path="$VISION_ENCODER_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --train_batch_size=1 \
  --sample_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --max_train_steps=50000 \
  --checkpointing_period=1000 \
  --sample_period=1000 \
  --checkpoints_total_limit=10 \
  --lr_scheduler="constant" \
  --learning_rate=1e-4 \
  --mixed_precision="bf16" \
  --dataloader_num_workers=4 \
  --image_aug \
  --dataset_type="finetune" \
  --state_noise_snr=40 \
  --report_to=wandb
  # NOTE: do NOT include --load_from_hdf5 for Open-X TFRecords
