# for temp in 4 8 10000; do

REMASKATOR_CHECKPOINT_PATH=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/wall-x-lerobot/checkpoints/remaskator_for_vae.pth
CHECKPOINT_PATH=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/wall-x-lerobot/checkpoints/mdlm_dit.pth

ROOT_RESULTS_DIR=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/results/vae_with_accuracy/nucleus_10/remaskator
mkdir -p ${ROOT_RESULTS_DIR}

# for temp in 0; do
for temp in 0 0.5 1; do
  mkdir -p ${ROOT_RESULTS_DIR}/temp_${temp}
  echo "Sampling with temperature ${temp}"
  CUDA_VISIBLE_DEVICES=2 HF_TOKEN="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" python main.py \
    mode=sample_eval \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    sampling.num_sample_batches=16 \
    sampling.steps=128 \
    data.wrap=False \
    data=openwebtext-split \
    parameterization=subs \
    backbone=dit \
    model.length=128 \
    model.cond_dim_embedding=384 \
    seed=11 \
    sampling.predictor=remaskator \
    sampling.remaskator_temperature=${temp} \
    sampling.remaskator_t_off=0.05 \
    sampling.remaskator_t_on=0.55 \
    sampling.sample_embeddings_from=validation \
    sampling.remaskator_checkpoint_path=${REMASKATOR_CHECKPOINT_PATH} \
    eval.checkpoint_path=${CHECKPOINT_PATH} \
    text_embedder.use_text_embedder=True \
    text_embedder.use_condition_during_sampling_until=1.0 \
    text_embedder.embedding_ema_decay=0.0 \
    text_embedder.num_embedding_updates=0 \
    text_embedder.model_name=sentence-transformers/all-MiniLM-L6-v2 \
    text_embedder.cond_dropout=0.0 \
    text_embedder.random_projection_dim=null \
    text_embedder.noise=0.0 \
    remaskator.use_residual_modulation=false \
    remaskator.use_weighted_sum=false \
    remaskator.global_conditioning=true \
    sub_conditioning=disabled \
    vae_encoder.enabled=true \
    hydra.run.dir=${ROOT_RESULTS_DIR}/temp_${temp} \
    sampling.nucleus_p=1.0  > ${ROOT_RESULTS_DIR}/temp_${temp}/sample.log 2>&1
done
    # sampling.gaussian_checkpoint_path=/mnt/virtual_ai0001071-01239_SR006-nfs2/afedorov/projects/mdlm-fork/subdiffusion_train/checkpoints/best.ckpt \
