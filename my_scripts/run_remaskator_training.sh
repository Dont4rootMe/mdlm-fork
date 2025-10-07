#!/bin/bash

# Training script for remaskator using Hydra parameter overrides
# Uses the specified checkpoint with 5 epochs and batch size 256

checkpoint_path=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/wall-x-lerobot/checkpoints/dit_pos_embedding_cond.pth
save_dir=/mnt/virtual_ai0001071-01239_SR006-nfs1/afedorov/projects/mdlm-fork/remaskator_train_vae_embed_attention

echo "Starting remaskator training..."
echo "Checkpoint: $checkpoint_path"
echo "Save directory: $save_dir"

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7

# Run training with Hydra parameter overrides
# WANDB_API_KEY=e54e11c5a3971ce143232dc777a77b7734c1d25e \
# WANDB_BASE_URL=https://api.wandb.ai \
python remaskator_train.py \
  TYPE_OF_CONDITIONING='pos_embedding' \
  model=small \
  mode=train \
  eval.checkpoint_path=$checkpoint_path \
  checkpointing.save_dir=$save_dir \
  experiment_name='remaskator_train_vae_embed_pos_embedding' \
  trainer.max_epochs=10 \
  loader.batch_size=128 \
  loader.eval_batch_size=128 \
  trainer.accumulate_grad_batches=1 \
  loader.global_batch_size=896 \
  trainer.precision=bf16 \
  +trainer.strategy=ddp_find_unused_parameters_true \
  trainer.val_check_interval=5000 \
  vae_encoder.enabled=true \
  sub_conditioning.enabled=false \
  model.length=128 \
  data.wrap=false \
  data=openwebtext-split \
  callbacks.checkpoint_monitor.monitor=val/loss \
  remaskator.use_residual_modulation=false \
  remaskator.use_weighted_sum=false \
  remaskator.global_conditioning=true \
  remaskator.initialization=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/wall-x-lerobot/checkpoints/mdlm_dit.pth

echo "Training completed!"
