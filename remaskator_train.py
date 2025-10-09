import os
import math
import json
import time
from typing import Optional, Tuple, Dict, Any
from collections import defaultdict

import hydra
import lightning as L
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from lightning.pytorch.callbacks import LearningRateMonitor
from main import _print_config

import dataloader
import diffusion as diffusion_mod
from models.remaskator import RemaskatorNet
from models.remaskator2 import Remaskator2Net


class JSONLogger:
    """Custom JSON logger that saves metrics to disk and creates dynamic plots."""
    
    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(save_dir, f"{experiment_name}_metrics.json")
        self.plot_dir = os.path.join(save_dir, "plots")
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = defaultdict(list)
        self.step_count = 0
        
        # Initialize plots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle(f'Training Metrics - {experiment_name}')
        plt.tight_layout()
        
        # Configure subplots
        self.axes[0, 0].set_title('Loss')
        self.axes[0, 0].set_xlabel('Step')
        self.axes[0, 0].set_ylabel('Loss')
        
        self.axes[0, 1].set_title('Accuracy')
        self.axes[0, 1].set_xlabel('Step')
        self.axes[0, 1].set_ylabel('Accuracy')
        
        self.axes[0, 2].set_title('Denoiser Mistake Ratio')
        self.axes[0, 2].set_xlabel('Step')
        self.axes[0, 2].set_ylabel('Mistake Ratio')
        
        self.axes[1, 0].set_title('Precision')
        self.axes[1, 0].set_xlabel('Step')
        self.axes[1, 0].set_ylabel('Precision')
        
        self.axes[1, 1].set_title('Recall')
        self.axes[1, 1].set_xlabel('Step')
        self.axes[1, 1].set_ylabel('Recall')
        
        self.axes[1, 2].set_title('F1 Score')
        self.axes[1, 2].set_xlabel('Step')
        self.axes[1, 2].set_ylabel('F1')
        
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = 'train'):
        """Log metrics to JSON file and update plots."""
        timestamp = time.time()
        
        # Store metrics with metadata
        log_entry = {
            'step': step,
            'phase': phase,
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        # Append to JSON file
        with open(self.log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
        
        # Update internal storage for plotting
        for key, value in metrics.items():
            metric_key = f"{phase}_{key}"
            self.metrics[metric_key].append((step, value))
        
        # Update plots every 10 steps to avoid too frequent updates
        if step % 10 == 0:
            try:
                self._update_plots()
            except Exception as e:
                print(f"Warning: Failed to update plots at step {step}: {e}")
    
    def _update_plots(self):
        """Update matplotlib plots with current metrics."""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot metrics
        metrics_to_plot = [
            ('loss', self.axes[0, 0]),
            ('acc', self.axes[0, 1]), 
            ('denoiser_mistake_ratio', self.axes[0, 2]),
            ('precision', self.axes[1, 0]),
            ('recall', self.axes[1, 1]),
            ('f1', self.axes[1, 2])
        ]
        
        for metric_name, ax in metrics_to_plot:
            train_key = f"train_{metric_name}"
            val_key = f"val_{metric_name}"
            
            # Plot training metrics
            if train_key in self.metrics and len(self.metrics[train_key]) > 0:
                steps, values = zip(*self.metrics[train_key])
                ax.plot(steps, values, label='Train', color='blue', alpha=0.7)
            
            # Plot validation metrics
            if val_key in self.metrics and len(self.metrics[val_key]) > 0:
                steps, values = zip(*self.metrics[val_key])
                ax.plot(steps, values, label='Validation', color='red', alpha=0.7)
            
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plot_dir, f"training_metrics.png")
        self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        
    def finalize(self):
        """Finalize logging and save final plots."""
        self._update_plots()
        final_plot_path = os.path.join(self.plot_dir, f"final_training_metrics.png")
        self.fig.savefig(final_plot_path, dpi=300, bbox_inches='tight')
        plt.close(self.fig)


class RemaskatorModule(L.LightningModule):
  """Lightning module that trains RemaskatorNet against a frozen diffusion denoiser.

  Training step:
    - Sample x0 from train batch
    - Sample t ~ Uniform(eps, 1), s ~ Uniform(0, t - eps)
    - Compute x_t via q(x_t | x0, t)
    - Compute x_s via a single DDPM update from t to s conditioned on embedding(x0)
    - For tokens newly generated between x_t and x_s, label y=1 if x_s!=x0 (should remask), else 0
    - Optimize BCE on those positions only
  """

  def __init__(self, config: omegaconf.DictConfig, tokenizer, denoiser: diffusion_mod.Diffusion, json_loggers: Dict[str, JSONLogger] = None):
    super().__init__()
    
    self.index_first_val = True
    
    self.save_hyperparameters(ignore=['tokenizer', 'denoiser', 'json_logger'])
    self.config = config
    self.tokenizer = tokenizer
    self.denoiser = denoiser.eval()
    self.json_loggers = json_loggers
    for p in self.denoiser.parameters():
      p.requires_grad = False

    # Common attributes from denoiser
    self.mask_index = int(self.denoiser.mask_index)
    self.vocab_size = int(self.denoiser.vocab_size)
    self.seq_len = int(self.config.model.length)
    self.eps = float(self.config.training.sampling_eps)

    # Get remaskator-specific configuration
    use_residual_modulation = False
    use_weighted_sum = False
    if hasattr(self.config, 'remaskator'):
      use_residual_modulation = self.config.remaskator.get('use_residual_modulation', False)
      use_weighted_sum = self.config.remaskator.get('use_weighted_sum', False)

    self.net = Remaskator2Net(
      vocab_size=self.vocab_size,
      config=self.config,
      cond_dim=self.denoiser.cond_dim,
      use_residual_modulation=use_residual_modulation,
      use_weighted_sum=use_weighted_sum,
    )

    # Get learning rate from remaskator config or fall back to optim config
    if hasattr(self.config, 'remaskator') and hasattr(self.config.remaskator, 'lr'):
      self.lr = float(self.config.remaskator.lr)
    else:
      self.lr = float(self.config.optim.lr)

    # All metrics computed ad-hoc per step on masked positions

  def _is_remaskator_residual_modulation_enabled(self):
    """Check if remaskator uses residual modulation."""
    if hasattr(self.config, 'remaskator'):
      return self.config.remaskator.get('use_residual_modulation', False)
    return False

  @torch.no_grad()
  def _sample_t_s(self, batch_size: int, device: torch.device) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    # t ~ Uniform(eps, 1), s ~ Uniform(0, t - eps), dt = t - s
    eps = torch.tensor(self.eps, device=device, dtype=torch.float32)
    t = (1 - eps) * torch.rand(batch_size, device=device) + eps
    # Ensure s <= t - eps
    # Sample s ~ Uniform(t-0.1, t-eps), clamped to [0, t-eps]
    # s_min = torch.clamp(t - 0.1, min=0.0)
    # s_max = torch.clamp(t - eps, min=0.0)
    # s = s_min + torch.rand_like(t) * (s_max - s_min)

    s = 0

    dt = t - s
    return t, s, dt

  @torch.no_grad()
  def _compute_xt_xs_and_labels(self, x0: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor]:
    device = x0.device
    batch_size = x0.shape[0]

    # Sample times
    t, s, dt = self._sample_t_s(batch_size, device)
    t_in = t.view(batch_size, 1)
    dt_in = dt.view(batch_size, 1)

    # Compute q(x_t | x0)
    sigma_t, _ = self.denoiser.noise(t)
    move_chance = 1.0 - torch.exp(-sigma_t)
    if move_chance.ndim == 1:
      move_chance = move_chance[:, None]
    xt = self.denoiser.q_xt(x0, move_chance)

    # Conditioning on x0 embedding if available
    if self.config.remaskator.get('global_conditioning', False):
      cond = self.denoiser.indices_to_text_embeddings(x0)
    else:
      cond = None

    # One-step DDPM update from t to s
    xs = self.denoiser._ddpm_update(xt, t_in, dt_in, condition=cond)

    # Identify newly generated tokens between xt and xs
    new_mask = (xt == self.mask_index) & (xs != self.mask_index)

    # Targets: 1 if incorrect (should remask), 0 if correct (keep)
    target = (xs != x0).to(torch.float32)

    return xt, xs, new_mask.to(torch.bool), target

  def _compute_loss_and_metrics(self, logits: torch.FloatTensor, target: torch.FloatTensor, new_mask: torch.BoolTensor):
    # BCE over all tokens (no masking or class balancing)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')

    with torch.no_grad():
      preds = (torch.sigmoid(logits) > 0.5).to(torch.int32)
      true = target.to(torch.int32)

      total = preds.numel()
      if total == 0:
        acc = torch.tensor(0.0, device=logits.device)
        precision = torch.tensor(0.0, device=logits.device)
        recall = torch.tensor(0.0, device=logits.device)
        f1 = torch.tensor(0.0, device=logits.device)
        mistake_ratio = torch.tensor(0.0, device=logits.device)
      else:
        # Accuracy over all tokens
        correct = (preds == true).sum()
        acc = correct.float() / total

        # Precision, Recall, F1 over all tokens
        tp = ((preds == 1) & (true == 1)).sum().float()
        fp = ((preds == 1) & (true == 0)).sum().float()
        fn = ((preds == 0) & (true == 1)).sum().float()

        precision = tp / (tp + fp).clamp(min=1e-8)
        recall = tp / (tp + fn).clamp(min=1e-8)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

        # Ratio of denoiser mistakes among all tokens
        mistakes = (true == 1).sum().float()
        mistake_ratio = mistakes / total
      
    return loss, acc, mistake_ratio, precision, recall, f1

  def forward(self, x_tokens: torch.LongTensor, cond: Optional[torch.FloatTensor] = None, curr_embed: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
    return self.net(x_tokens, cond, curr_embed)

  def training_step(self, batch, batch_idx):
    self.index_first_val = True
    
    
    x0 = batch['input_ids']  # (batch, seq_len)
    x0 = x0.to(self.device)

    # Compute xs and labels
    xt, xs, new_mask, target = self._compute_xt_xs_and_labels(x0)
    
    if self._is_remaskator_residual_modulation_enabled():
      curr_embed = self.denoiser.indices_to_text_embeddings(xs)
    else:
      curr_embed = None

    # Conditioning on x0 embedding if available
    if self.config.remaskator.get('global_conditioning', False):
      cond = self.denoiser.indices_to_text_embeddings(x0)
    else:
      cond = None

    logits = self.forward(xs, cond, curr_embed)

    loss, acc, mistake_ratio, precision, recall, f1 = self._compute_loss_and_metrics(logits, target, new_mask)
    
    # Log metrics to Lightning (for progress bar)
    # self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    # self.log('train/acc', acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    # self.log('train/denoiser_mistake_ratio', mistake_ratio, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    # self.log('train/precision', precision, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    # self.log('train/recall', recall, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    # self.log('train/f1', f1, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
    
    # Log metrics to JSON logger and update plots
    if self.json_loggers['train'] is not None:
      metrics = {
        'loss': loss.item(),
        'acc': acc.item(),
        'denoiser_mistake_ratio': mistake_ratio.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
      }
      self.json_loggers['train'].log_metrics(metrics, self.global_step, 'train')
    
    return loss

  def validation_step(self, batch, batch_idx):
    if self.index_first_val:
      save_dir = f"/mnt/virtual_ai0001071-01239_SR006-nfs1/afedorov/projects/mdlm-fork/remaskator_checkpoints/remaskator_attention_conditioning"
      os.makedirs(save_dir, exist_ok=True)
      torch.save(self.net.state_dict(), f"{save_dir}/net_state.pt")
    self.index_first_val = False
    
    x0 = batch['input_ids'].to(self.device)
    xt, xs, new_mask, target = self._compute_xt_xs_and_labels(x0)
    if self._is_remaskator_residual_modulation_enabled():
      curr_embed = self.denoiser.indices_to_text_embeddings(xs)
    else:
      curr_embed = None
    if self.config.remaskator.get('global_conditioning', False):
      cond = self.denoiser.indices_to_text_embeddings(x0)
    else:
      cond = None
    logits = self.forward(xs, cond, curr_embed)

    loss, acc, mistake_ratio, precision, recall, f1 = self._compute_loss_and_metrics(logits, target, new_mask)
    
    # Log metrics to Lightning (for progress bar)
    # self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    # self.log('val/acc', acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    # self.log('val/denoiser_mistake_ratio', mistake_ratio, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    # self.log('val/precision', precision, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    # self.log('val/recall', recall, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    # self.log('val/f1', f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
    # Log metrics to JSON logger and update plots
    if self.json_loggers['val'] is not None:
      metrics = {
        'loss': loss.item(),
        'acc': acc.item(),
        'denoiser_mistake_ratio': mistake_ratio.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
      }
      self.json_loggers['val'].log_metrics(metrics, self.global_step, 'val')
    
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(self.config.optim.beta1, self.config.optim.beta2), eps=self.config.optim.eps, weight_decay=self.config.optim.weight_decay)
    scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
    return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'trainer/lr'}]

  def on_train_end(self):
    """Called when training ends - finalize JSON logger."""
    for logger in self.json_loggers.values():
      logger.finalize()
      print(f"Training completed. Metrics saved to: {logger.log_file}")
      print(f"Plots saved to: {logger.plot_dir}")

    if self.json_loggers['train'] is not None:
      self.json_loggers['train'].finalize()
      print(f"Training completed. Metrics saved to: {self.json_logger['train'].log_file}")
      print(f"Plots saved to: {self.json_logger['train'].plot_dir}")
    if self.json_loggers['val'] is not None:
      self.json_logger['val'].finalize()
      print(f"Validation completed. Metrics saved to: {self.json_loggers['val'].log_file}")
      print(f"Plots saved to: {self.json_logger['val'].plot_dir}")


def _load_denoiser(config: omegaconf.DictConfig, tokenizer) -> diffusion_mod.Diffusion:
  # if 'hf' in config.backbone:
  #   return diffusion_mod.Diffusion(config, tokenizer=tokenizer).to('cuda')
  policy = diffusion_mod.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
  )
  return policy

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config: omegaconf.DictConfig):

  _print_config(config, resolve=True, save_cfg=True)

  # Tokenizer and dataloaders
  tokenizer = dataloader.get_tokenizer(config)
  train_loader, valid_loader = dataloader.get_dataloaders(config, tokenizer)

  # Denoiser (frozen)
  assert config.eval.checkpoint_path, 'Please set eval.checkpoint_path to the denoiser checkpoint.'
  denoiser = _load_denoiser(config, tokenizer)
  denoiser.eval()
  for p in denoiser.parameters():
    p.requires_grad = False

  # Setup JSON logger
  experiment_name = f"remaskator_{config.wandb.name if hasattr(config, 'wandb') and hasattr(config.wandb, 'name') else 'experiment'}"
  json_loggers = {
    'train': JSONLogger(
      save_dir=config.checkpointing.save_dir,
      experiment_name=experiment_name + '_train'
    ),
    'val': JSONLogger(
      save_dir=config.checkpointing.save_dir + '_val',
      experiment_name=experiment_name + '_val'
    )
  }
  print(f"JSON logger initialized:")
  print(f"  Metrics will be saved to: {json_loggers['train'].log_file} and {json_loggers['val'].log_file}")
  print(f"  Plots will be saved to: {json_loggers['train'].plot_dir} and {json_loggers['val'].plot_dir}")

  # Module
  module = RemaskatorModule(config=config, tokenizer=tokenizer, denoiser=denoiser, json_loggers=json_loggers)
  # load weight from denoiser
  if config.remaskator.initialization is not None:
    
    state_dict = torch.load(config.remaskator.initialization)
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict['state_dict'].items() if 'backbone' in k}
    
    missing_keys, unexpected_keys = module.net.dit.load_state_dict(state_dict, strict=False)
  else: 
    missing_keys, unexpected_keys = module.net.dit.load_state_dict(denoiser.backbone.state_dict(), strict=False)
  print(f"Missing keys: {missing_keys}")
  print(f"Unexpected keys: {unexpected_keys}")
  module.net.change_final_layer()

  # Setup callbacks
  lr_monitor = LearningRateMonitor(logging_interval='step')
  callbacks = [lr_monitor]
  
  # Add any existing callbacks from config
  if hasattr(config, 'callbacks') and config.callbacks:
    existing_callbacks = [hydra.utils.instantiate(cb) for cb in config.callbacks.values()]
    callbacks.extend(existing_callbacks)

  # Trainer (no logger needed since we use JSON logger directly)
  trainer: L.Trainer = hydra.utils.instantiate(config.trainer)

  # Fit
  trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

  # Finalize JSON logger (save final plots)
  for logger in json_loggers.values():
    logger.finalize()
    print(f"{logger.experiment_name} completed. Metrics saved to: {logger.log_file}")
    print(f"Plots saved to: {logger.plot_dir}")

  # Save final checkpoint
  ckpt_dir = os.path.join(config.checkpointing.save_dir, 'checkpoints_remaskator')
  os.makedirs(ckpt_dir, exist_ok=True)
  trainer.save_checkpoint(os.path.join(ckpt_dir, 'last.ckpt'))


if __name__ == '__main__':
  main()

