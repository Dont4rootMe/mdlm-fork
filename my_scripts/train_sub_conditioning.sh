#!/bin/bash

# Clean training script for sub conditioning
# Uses the new configuration structure

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main.py \
  model=small \
  data=openwebtext-split \
  parameterization=subs \
  model.length=128 \
  model.cond_dim_embedding=384 \
  eval.compute_generative_perplexity=True \
  sampling.num_sample_batches=4 \
  sampling.steps=128 \
  +sampling.remdm_mode=null \
  checkpointing.resume_from_ckpt=False \
  checkpointing.save_dir=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/output_sub_conditioning/ \
  loader.global_batch_size=512 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  trainer.val_check_interval=500 \
  trainer.devices=8 \
  trainer.num_nodes=1 \
  trainer.max_steps=100000 \
  trainer.max_epochs=-1 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  text_embedder.use_text_embedder=True \
  text_embedder.model_name=sentence-transformers/all-MiniLM-L6-v2 \
  text_embedder.cond_dropout=0.0 \
  text_embedder.random_projection_dim=null \
  text_embedder.noise=0.1 \
  eval.generate_samples=True \
  +embedding_cache_dir=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/embedding_cache \
  data.wrap=False \
  sub_conditioning=enabled
