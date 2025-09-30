import os

import hydra
import lightning as L
import omegaconf

import dataloader
import utils
import embeddings as emb
from models.embedding_diffusion_module import EmbeddingDiffusionModule
from main import _print_config
from lightning_json_logger import LightningJSONLogger


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)

  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if not config.text_embedder.use_text_embedder:
    raise ValueError('Enable text_embedder.use_text_embedder to train embedding diffusion.')

  text_embedder = emb.TextEmbedder(
    model_name=config.text_embedder.model_name,
    random_projection_dim=config.text_embedder.random_projection_dim)

  # Data
  train_loader, valid_loader = dataloader.get_dataloaders(
    config, tokenizer, skip_train=False, skip_valid=False, valid_seed=config.seed)

  # Module
  module = EmbeddingDiffusionModule(
    cond_dim=int(text_embedder.cond_dim),
    timesteps=int(config.embedding_diffusion.timesteps),
    hidden_dim=int(config.embedding_diffusion.hidden_dim),
    num_layers=int(config.embedding_diffusion.num_layers),
    t_sampling_exponent=float(getattr(config.embedding_diffusion, 't_sampling_exponent', 2.0)),
    lr=float(config.optim.lr),
    weight_decay=float(config.optim.weight_decay),
    length_mean=None,
    length_std=None,
    max_seq_len=int(config.model.length),
    tokenizer=tokenizer,
    text_embedder=text_embedder,
    fid_sample_size=int(config.embedding_diffusion.fid_sample_size),
  )

  # Lightning Trainer - Use JSON logger by default
  experiment_name = f"embedding_diffusion_{config.wandb.name if hasattr(config, 'wandb') and hasattr(config.wandb, 'name') else 'experiment'}"
  
  # Check if explicitly configured to use wandb
  use_wandb = config.get('logging', {}).get('use_wandb_logger', False)
  
  if use_wandb and config.get('wandb', None) is not None:
    # Use wandb logger if explicitly requested
    json_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
    print(f"WandB logger initialized for experiment: {experiment_name}")
  else:
    # Default to JSON logger
    print('\n\n\n', config.checkpointing.save_dir, '\n\n\n')
    json_logger = LightningJSONLogger(
      save_dir=config.checkpointing.save_dir,
      experiment_name=experiment_name
    )
    print(f"JSON logger initialized:")
    print(f"  Experiment directory: {json_logger.log_dir}")
    print(f"  Train metrics: {json_logger._train_log_file}")
    print(f"  Val metrics: {json_logger._val_log_file}")
    print(f"  Plots directory: {json_logger._plot_dir}")
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=json_logger)

  trainer.fit(module, train_loader, valid_loader)

  # Finalize JSON logger (save final plots)
  if json_logger is not None and hasattr(json_logger, 'finalize'):
    json_logger.finalize()
    print(f"Training completed. Metrics saved to: {json_logger._train_log_file} and {json_logger._val_log_file}")
    print(f"Plots saved to: {json_logger._plot_dir}")

  # Save checkpoint
  ckpt_path = getattr(config.embedding_diffusion, 'checkpoint_path', './checkpoints/embedding_diffusion.ckpt')
  os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
  trainer.save_checkpoint(ckpt_path)
  print(f"Saved checkpoint to {ckpt_path}")


if __name__ == '__main__':
  main()

 