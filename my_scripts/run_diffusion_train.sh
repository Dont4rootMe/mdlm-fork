#!/bin/bash

# Training script for remaskator using Hydra parameter overrides
# Uses the specified checkpoint with 5 epochs and batch size 256

# checkpoint_path=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/weights/all_weights/ft_small_from_ckpt_small_cond_no_wrap_seqlen128_cond_dropout0.0/checkpoints/4-125000.ckpt
# save_dir=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/output/remaskator_cond_no_wrap_seqlen128_cond_dropout0.0_rpnull_noise0.0_use-modulation_uweighted-sum
# wandb_name=remaskator-cond-no-wrap-seqlen128-cond-dropout0.0-rpnull_noise0.0-$(date +%Y%m%d_%H%M%S)

# echo "Starting remaskator training..."
# echo "Checkpoint: $checkpoint_path"
# echo "Save directory: $save_dir"
# echo "Wandb name: $wandb_name"


# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=7

# Run training with Hydra parameter overrides
# WANDB_API_KEY=e54e11c5a3971ce143232dc777a77b7734c1d25e \
# WANDB_BASE_URL=https://api.wandb.ai \
python train_embedding_diffusion.py \
  +trainer.max_epochs=10 \
  loader.batch_size=384 \
  loader.eval_batch_size=384 \
  trainer.accumulate_grad_batches=1 \
  loader.global_batch_size=384 \
  trainer.precision=bf16 \
  trainer.val_check_interval=1000 \
  text_embedder.use_text_embedder=true \
  text_embedder.model_name=sentence-transformers/all-MiniLM-L6-v2 \
  text_embedder.noise=0.0 \
  model.length=128 \
  data.wrap=false \
  data=openwebtext-split \
  optim.lr=5e-5 \
  callbacks.checkpoint_monitor.monitor=val/loss \
  +use_residual_modulation=true \
  +use_weighted_sum=true

echo "Training completed!"