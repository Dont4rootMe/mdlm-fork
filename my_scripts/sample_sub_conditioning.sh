#!/bin/bash

# Sampling script for sub conditioning evaluation
# Uses the new configuration structure

CUDA_VISIBLE_DEVICES=2 \
python main.py \
  mode=sample_eval \
  model=small \
  data=openwebtext-split \
  parameterization=subs \
  model.length=128 \
  model.cond_dim_embedding=384 \
  sampling.num_sample_batches=10 \
  sampling.steps=128 \
  sampling.predictor=ddpm_cache \
  loader.eval_batch_size=32 \
  text_embedder.use_text_embedder=True \
  text_embedder.model_name=sentence-transformers/all-MiniLM-L6-v2 \
  text_embedder.cond_dropout=0.0 \
  text_embedder.random_projection_dim=null \
  text_embedder.noise=0.0 \
  +embedding_cache_dir=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/embedding_cache \
  data.wrap=False \
  sub_conditioning=enabled \
  eval.checkpoint_path=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/output_sub_conditioning/checkpoints/best.ckpt \
  eval.compute_generative_perplexity=True \
  eval.generate_samples=True \
  eval.save_sampling_trajectory=True \
  eval.max_trajectories_to_save=5
