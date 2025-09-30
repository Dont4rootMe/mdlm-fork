#!/bin/bash

# Test script to verify sub conditioning is working
# Tests the dependence of model output on condition embeddings

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
  mode=condition_dependence_test \
  model=small \
  data=openwebtext-split \
  parameterization=subs \
  model.length=128 \
  model.cond_dim_embedding=384 \
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
  eval.disable_ema=False
