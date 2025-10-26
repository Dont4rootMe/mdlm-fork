import itertools
import math
import os
import json
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils
import embeddings
from models.remaskator import RemaskatorNet
from models.remaskator2 import Remaskator2Net

LOG2 = math.log(2)


def _compute_levenshtein_distance(seq1, seq2):
  """Compute Levenshtein distance between two sequences (token lists).
  
  Args:
    seq1: First sequence (list of tokens)
    seq2: Second sequence (list of tokens)
    
  Returns:
    int: Levenshtein distance between the sequences
  """
  if len(seq1) == 0:
    return len(seq2)
  if len(seq2) == 0:
    return len(seq1)
    
  # Create a matrix to store distances
  matrix = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
  
  # Initialize first row and column
  for i in range(len(seq1) + 1):
    matrix[i][0] = i
  for j in range(len(seq2) + 1):
    matrix[0][j] = j
    
  # Fill the matrix
  for i in range(1, len(seq1) + 1):
    for j in range(1, len(seq2) + 1):
      if seq1[i-1] == seq2[j-1]:
        cost = 0
      else:
        cost = 1
        
      matrix[i][j] = min(
        matrix[i-1][j] + 1,      # deletion
        matrix[i][j-1] + 1,      # insertion
        matrix[i-1][j-1] + cost  # substitution
      )
  
  return matrix[len(seq1)][len(seq2)]


def _compute_text_based_accuracy_and_levenshtein(tokenizer, predicted_tokens, reference_tokens):
  """Compute accuracy and Levenshtein distance by comparing decoded texts.
  
  This is more accurate for VAE decoder as it accounts for tokenization differences.
  
  Args:
    tokenizer: The tokenizer used to decode tokens
    predicted_tokens: Tensor of shape (batch_size, seq_len) with predicted token ids
    reference_tokens: Tensor of shape (batch_size, seq_len) with reference token ids
    
  Returns:
    tuple: (mean_accuracy, mean_levenshtein_distance) across batch
  """
  if predicted_tokens.shape != reference_tokens.shape:
    return 0.0, 0.0
    
  batch_size = predicted_tokens.shape[0]
  total_accuracy = 0.0
  total_levenshtein = 0.0
  valid_pairs = 0
  
  for i in range(batch_size):
    pred_tokens = predicted_tokens[i].cpu().tolist()
    ref_tokens = reference_tokens[i].cpu().tolist()
    
    # Decode to text and re-tokenize for fair comparison
    try:
      pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
      ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
      
      # Re-tokenize both texts to get clean token sequences
      pred_clean = tokenizer.encode(pred_text, add_special_tokens=False)
      ref_clean = tokenizer.encode(ref_text, add_special_tokens=False)
      
      if len(ref_clean) == 0:
        continue
        
      # Compute accuracy on clean tokens
      min_length = min(len(pred_clean), len(ref_clean))
      matches = sum(1 for j in range(min_length) if pred_clean[j] == ref_clean[j])
      
      # Penalize length differences
      length_penalty = abs(len(pred_clean) - len(ref_clean))
      total_positions = max(len(pred_clean), len(ref_clean))
      pair_accuracy = (matches - length_penalty) / total_positions if total_positions > 0 else 0.0
      pair_accuracy = max(0.0, pair_accuracy)  # Ensure non-negative
      
      # Compute Levenshtein distance on clean tokens
      levenshtein_dist = _compute_levenshtein_distance(pred_clean, ref_clean)
      normalized_levenshtein = levenshtein_dist / len(ref_clean) if len(ref_clean) > 0 else 0.0
      
      total_accuracy += pair_accuracy
      total_levenshtein += normalized_levenshtein
      valid_pairs += 1
      
    except Exception:
      # Fallback to token-based comparison if decoding fails
      continue
  
  if valid_pairs > 0:
    return total_accuracy / valid_pairs, total_levenshtein / valid_pairs
  else:
    return 0.0, 0.0


def _compute_first_step_accuracy_and_levenshtein(tokenizer, predicted_tokens, reference_tokens):
  """Compute token-level accuracy and Levenshtein distance for first step predictions.
  
  Args:
    tokenizer: The tokenizer used to encode/decode texts
    predicted_tokens: Tensor of shape (batch_size, seq_len) with predicted token ids
    reference_tokens: Tensor of shape (batch_size, seq_len) with reference token ids
    
  Returns:
    tuple: (mean_accuracy, mean_levenshtein_distance) across batch
  """
  if predicted_tokens.shape != reference_tokens.shape:
    return 0.0, 0.0
    
  batch_size = predicted_tokens.shape[0]
  total_accuracy = 0.0
  total_levenshtein = 0.0
  valid_pairs = 0
  
  for i in range(batch_size):
    pred_tokens = predicted_tokens[i].cpu().tolist()
    ref_tokens = reference_tokens[i].cpu().tolist()
    
    # Skip padding tokens if present
    pad_token_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_token_id is not None:
      # Find first pad token in reference to get actual length
      try:
        ref_len = ref_tokens.index(pad_token_id)
        ref_tokens = ref_tokens[:ref_len]
        pred_tokens = pred_tokens[:ref_len]
      except ValueError:
        # No pad token found, use full length
        pass
    
    if len(ref_tokens) == 0:
      continue
      
    # Compute token-level accuracy for this pair
    if len(ref_tokens) > 0:
      # Pad shorter sequence with a special "mismatch" token to ensure fair comparison
      max_length = max(len(pred_tokens), len(ref_tokens))
      
      # Extend sequences to same length (shorter one gets mismatches)
      pred_extended = pred_tokens + [-1] * (max_length - len(pred_tokens))
      ref_extended = ref_tokens + [-1] * (max_length - len(ref_tokens))
      
      # Count exact matches at each position
      matches = sum(1 for j in range(max_length) if pred_extended[j] == ref_extended[j])
      pair_accuracy = matches / len(ref_tokens)  # Normalize by reference length
    else:
      pair_accuracy = 0.0
    
    # Compute Levenshtein distance
    levenshtein_dist = _compute_levenshtein_distance(pred_tokens, ref_tokens)
    # Normalize by reference length to get a relative distance
    normalized_levenshtein = levenshtein_dist / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    total_accuracy += pair_accuracy
    total_levenshtein += normalized_levenshtein
    valid_pairs += 1
  
  # Return mean accuracy and mean normalized Levenshtein distance across batch
  if valid_pairs > 0:
    return total_accuracy / valid_pairs, total_levenshtein / valid_pairs
  else:
    return 0.0, 0.0


def _sample_categorical(categorical_probs):
  categorical_probs = categorical_probs.to(torch.float64)
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class FirstStepAccuracy(torchmetrics.aggregation.MeanMetric):
  """Metric for tracking accuracy on first denoising step."""
  pass


class FirstStepLevenshtein(torchmetrics.aggregation.MeanMetric):
  """Metric for tracking Levenshtein distance on first denoising step."""
  pass


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    
    
    print('\n\n\n\nusing tokenizer: ', tokenizer, '\n\n\n\n')
    
    self.change_time_scheduler = self.config.training.change_scheduler

    self.tokenizer = tokenizer
    self.vocab_size = len(self.tokenizer)
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables

    # Choose embedding source based on config
    self.sample_embeddings_from = getattr(self.config.sampling, 'sample_embeddings_from', 'validation')

    if self.config.text_embedder.use_text_embedder:
      # Note: TextEmbedder is not an nn.Module to keep its parameters frozen
      # during training and exclude them from EMA. Device movement is handled
      # manually in training hooks.
      if self.config.vae_encoder.enabled:
        
        from vae_embeddings import VAEEncoder
        
        self.text_embedder = VAEEncoder(self.config.vae_encoder)
        self.cond_dim = self.config.vae_encoder.latent_encoder.latent.dim
        
      else:
        self.text_embedder = embeddings.TextEmbedder(
          model_name=self.config.text_embedder.model_name,
          random_projection_dim=self.config.text_embedder.random_projection_dim)
        self.cond_dim = self.text_embedder.cond_dim
    else:
      self.text_embedder = None
      self.cond_dim = None

    # Lazy dataset handles for on-the-fly conditioning
    self._train_text_dataset = None
    self._valid_text_dataset = None

    if self.config.vae_encoder.enabled:
      self.mask_index = self.tokenizer.mask_token_id
    else:
      if (not hasattr(self.tokenizer, 'mask_token')
          or self.tokenizer.mask_token is None):
        self.mask_index = self.vocab_size
        self.vocab_size += 1
      else:
        self.mask_index = self.tokenizer.mask_token_id
    
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      # Get sub conditioning parameters from config
      use_residual_modulation = False
      use_weighted_sum = False
      
      if hasattr(self.config, 'sub_conditioning'):
        use_weighted_sum = self.config.sub_conditioning.get('use_weighted_sum', False)
        use_residual_modulation = self.config.sub_conditioning.get('use_residual_modulation', False)
      
      
      # =========================================
      #        - ADDING NEW CONDITIONING -       
      # =========================================
      
      if self.config.TYPE_OF_CONDITIONING == 'attention':
        print("Using attention conditioning")
        self.backbone = models.dit_new_condition.DIT(
          self.config, vocab_size=self.vocab_size, cond_dim=self.cond_dim,
        )
      elif self.config.TYPE_OF_CONDITIONING == 'pos_embedding':
        print("Using pos_embedding conditioning")
        self.backbone = models.dit_positional_condition.DIT(
          self.config, vocab_size=self.vocab_size, cond_dim=self.cond_dim,
        )
      elif self.config.TYPE_OF_CONDITIONING == 'vae_implementation':
        print("Using vae implementation conditioning")
        self.backbone = models.dit_vae_condition.DIT(
          self.config, vocab_size=self.vocab_size, cond_dim=self.cond_dim,
        )
      elif self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
        print("Using original vae decoder conditioning")
        self.backbone = models.original_vae_decoder.Decoder(
          self.config, vocab_size=self.vocab_size, cond_dim=self.cond_dim,
          mask_token_id=self.mask_index
        )
      else:
        print("Using dit conditioning")
        self.backbone = models.dit.DIT(
          self.config, vocab_size=self.vocab_size, cond_dim=self.cond_dim,
          use_residual_modulation=use_residual_modulation,
          use_weighted_sum=use_weighted_sum
        )
      
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')
    
    # First step denoising metrics (only for validation)
    first_step_metrics = torchmetrics.MetricCollection({
      'first_step_accuracy': FirstStepAccuracy(),
      'first_step_levenshtein': FirstStepLevenshtein(),
    })
    first_step_metrics.set_dtype(torch.float64)
    self.valid_first_step_metrics = first_step_metrics.clone(prefix='val/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)

    # Optional: initialize Remaskator model for guided sampling
    self.remaskator = None
    # Read remaskator interval from sampling section
    self.remaskator_t_off = float(self.config.sampling.remaskator_t_off)
    self.remaskator_t_on = float(self.config.sampling.remaskator_t_on)
    assert self.remaskator_t_off < self.remaskator_t_on, "remaskator_t_off must be less than remaskator_t_on"
    if self.config.text_embedder.use_text_embedder and self.config.sampling.remaskator_checkpoint_path is not None:
      self.remaskator_temperature = float(self.config.sampling.remaskator_temperature)
      if self.sampler == 'remaskator':
        ckpt_path = str(self.config.sampling.remaskator_checkpoint_path)
        if len(ckpt_path) == 0 or not os.path.exists(ckpt_path):
          raise ValueError(f'Remaskator checkpoint path {ckpt_path} does not exist')

        # load RemaskatorModule and then extract the RemaskatorNet from it
        remaskator_module_state_dict = torch.load(ckpt_path, map_location='cpu')
        remaskator_net_state_dict = {k: v for k, v in remaskator_module_state_dict.items()}  
        # self.remaskator = RemaskatorNet(
        #   vocab_size=self.vocab_size,
        #   seq_len=int(self.config.model.length),
        #   hidden_size=int(self.config.model.hidden_size),
        #   num_layers=int(self.config.model.n_blocks),
        #   num_heads=int(self.config.model.n_heads),
        #   cond_dim=self.cond_dim,
        #   dropout=float(self.config.model.dropout)
        # )
        self.remaskator = Remaskator2Net(
          vocab_size=self.vocab_size,
          config=self.config,
          cond_dim=self.cond_dim,
          use_residual_modulation=self.config.remaskator.use_residual_modulation,
          use_weighted_sum=self.config.remaskator.use_weighted_sum
        )
        self.remaskator.change_final_layer()
        missing, unexpected = self.remaskator.load_state_dict(remaskator_net_state_dict)
        # print("missing", missing)
        # print("unexpected", unexpected)
        # 1 / 0

        # freeze the RemaskatorNet
        for p in self.remaskator.parameters():
          p.requires_grad = False
        self.remaskator.eval()
    if self.config.training.ema > 0:
      
      print('\n========', len([*self.backbone.parameters()]), len([*self.noise.parameters()]), '\n========\n\n\n')
      
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()
    # Counter for how many sampling trajectories have been saved so far
    self._trajectories_saved = 0

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def _is_sub_conditioning_enabled(self):
    """Check if sub conditioning (residual modulation) is enabled."""
    if hasattr(self.config, 'sub_conditioning'):
      return self.config.sub_conditioning.get('use_residual_modulation', False)
    elif hasattr(self.config, 'dit'):
      # Fallback to old dit-specific config for backward compatibility
      return self.config.dit.get('use_residual_modulation', False)
    return False

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def _move_text_embedder_to_device(self):
    """Move text embedder to the current device."""
    if self.text_embedder is not None:
      self.text_embedder.model = self.text_embedder.model.to(self.device)
      self.text_embedder.device = self.device

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    
    # Move text embedder to correct device
    self._move_text_embedder_to_device()
    
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, condition, curr_embed=None):
    """Returns log score."""

    # For original_vae_decoder, pass sigma=None and curr_embed=None
    if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      processed_sigma = None
      processed_curr_embed = None
    else:
      processed_sigma = self._process_sigma(sigma)
      processed_curr_embed = curr_embed
    
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, processed_sigma, condition, curr_embed=processed_curr_embed)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  @torch.no_grad()
  def _compute_first_step_metrics(self, x0):
    """Compute first step denoising accuracy and Levenshtein distance metrics."""
    if self.parameterization == 'ar':
      return  # Skip for autoregressive models
    
    batch_size = x0.shape[0]
    
    # Special handling for original_vae_decoder (single-step denoising from condition)
    if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      # For original_vae_decoder: single-step denoising from fully masked input
      t = torch.ones(batch_size, device=x0.device) * 0.999999
      condition = self.indices_to_text_embeddings(x0)  # Condition from ground truth
      
      # Create fully masked input (what decoder receives)
      xt = torch.full_like(x0, self.mask_index)
      unet_conditioning = None
      curr_embed = None
    else:
      # Standard multi-step diffusion: sample a high noise level for first step
      t = torch.ones(batch_size, device=x0.device) * 0.99
      
      # Get condition embeddings
      condition = None
      if self.config.text_embedder.use_text_embedder and self.text_embedder is not None:
        condition = self.indices_to_text_embeddings(x0)
      
      # Create noisy version (mostly masked)
      sigma, _ = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])
      xt = self.q_xt(x0, move_chance)
      
      # Get current embeddings for sub conditioning
      curr_embed = None
      if True:
        curr_embed = self.indices_to_text_embeddings(xt)
    
    # Predict x0 from xt
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(xt, unet_conditioning, condition, curr_embed=curr_embed)
    
    # Get predicted tokens (argmax)
    # For original_vae_decoder, use raw logits (no parameterization processing)
    if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      # VAE decoder returns raw logits, just take argmax
      predicted_tokens = logits.argmax(dim=-1)
    else:
      # Apply parameterization processing for other models
      if self.parameterization == 'subs':
        # For subs parameterization, apply the same processing as in forward
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
      elif self.parameterization == 'd3pm':
        if self.subs_masking:
          logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
      
      predicted_tokens = logits.argmax(dim=-1)
    
    # Compute metrics
    if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      # Use text-based comparison for VAE decoder (more accurate)
      accuracy, levenshtein = _compute_text_based_accuracy_and_levenshtein(
        self.tokenizer, predicted_tokens, x0)
    else:
      # Use token-based comparison for other models
      accuracy, levenshtein = _compute_first_step_accuracy_and_levenshtein(
        self.tokenizer, predicted_tokens, x0)
    
    # Update metrics individually
    self.valid_first_step_metrics['val/first_step_accuracy'].update(
      torch.tensor(accuracy, device=x0.device), 
      weight=torch.tensor(batch_size, device=x0.device)
    )
    self.valid_first_step_metrics['val/first_step_levenshtein'].update(
      torch.tensor(levenshtein, device=x0.device), 
      weight=torch.tensor(batch_size, device=x0.device)
    )
    
    # Log metrics
    self.log_dict(self.valid_first_step_metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='train/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    
    # Move text embedder to correct device
    self._move_text_embedder_to_device()
    
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='val')
    
    # Compute first step denoising metrics
    if 'input_ids' in batch:
      self._compute_first_step_metrics(batch['input_ids'])
    
    return loss

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      reference_texts = None
      all_first_step_accuracies = []
      all_first_step_levenshteins = []
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples, reference_texts, first_step_acc, first_step_lev = self._sample()
        # Collect first step metrics if available
        if first_step_acc is not None:
          all_first_step_accuracies.append(first_step_acc)
        if first_step_lev is not None:
          all_first_step_levenshteins.append(first_step_lev)
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      
      # Log aggregated first step metrics
      if all_first_step_accuracies:
        avg_first_step_acc = sum(all_first_step_accuracies) / len(all_first_step_accuracies)
        self.log('val/first_step_accuracy_sampling', avg_first_step_acc, on_epoch=True, on_step=False, sync_dist=True)
      if all_first_step_levenshteins:
        avg_first_step_lev = sum(all_first_step_levenshteins) / len(all_first_step_levenshteins)
        self.log('val/first_step_levenshtein_sampling', avg_first_step_lev, on_epoch=True, on_step=False, sync_dist=True)
      
      # Log generated samples to TensorBoard as text
      if self.trainer.global_rank == 0 and self.trainer.logger is not None:
        try:
          text_samples_to_log = text_samples[: self.config.sampling.num_sample_log]
          # Log each sample as text with TensorBoard
          samples_text = '\n\n---\n\n'.join([f"Sample {i+1}:\n{s}" for i, s in enumerate(text_samples_to_log)])
          self.trainer.logger.experiment.add_text(
            'generated_samples',
            samples_text,
            global_step=self.global_step
          )
          
          # Log input and output texts in 'texts' group
          if reference_texts is not None:
            reference_texts_to_log = reference_texts[: self.config.sampling.num_sample_log]
            generated_texts_to_log = text_samples[: self.config.sampling.num_sample_log]
            
            # Log input texts (encoder inputs)
            input_texts = '\n\n---\n\n'.join([f"Input {i+1}:\n{s if s is not None else '[No reference]'}" 
                                              for i, s in enumerate(reference_texts_to_log)])
            self.trainer.logger.experiment.add_text(
              'texts/input_texts',
              input_texts,
              global_step=self.global_step
            )
            
            # Log output texts (decoder outputs)
            output_texts = '\n\n---\n\n'.join([f"Output {i+1}:\n{s}" 
                                               for i, s in enumerate(generated_texts_to_log)])
            self.trainer.logger.experiment.add_text(
              'texts/output_texts',
              output_texts,
              global_step=self.global_step
            )
            
            # Log paired input-output for easy comparison
            paired_texts = '\n\n---\n\n'.join([
              f"Pair {i+1}:\nInput: {ref if ref is not None else '[No reference]'}\n - - - \nOutput: {gen}"
              for i, (ref, gen) in enumerate(zip(reference_texts_to_log, generated_texts_to_log))
            ])
            self.trainer.logger.experiment.add_text(
              'texts/paired_texts',
              paired_texts,
              global_step=self.global_step
            )
        except Exception as e:
          # Fallback if TensorBoard text logging fails
          print(f"Warning: Failed to log text samples to TensorBoard: {e}")
          # Still log the number of samples generated
          self.log('val/num_samples_generated', len(text_samples_to_log), on_epoch=True, on_step=False)
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'train/learning_rate',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims, lens=None):
    """
    Returns a tensor of shape batch_dims filled with self.mask_index.
    If lens is provided (should be a 1D array of sequence lengths, one per sample),
    for each sample in the batch, the first lens[i] tokens are set to self.mask_index,
    the rest are set to self.tokenizer.pad_token_id.
    """
    if lens is None or self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      return self.mask_index * torch.ones(
        *batch_dims, dtype=torch.int64)
    else:
      # lens should be a 1D array or tensor of shape (batch_size,)
      assert len(batch_dims) == 2 # batch_dims should be (batch_size, seq_len)

      lens = torch.as_tensor(lens, dtype=torch.int64)
      batch_size = lens.shape[0]
      seq_len = batch_dims[1]
      out = torch.full((batch_size, seq_len), self.tokenizer.pad_token_id, dtype=torch.int64, device=lens.device)
      for i, l in enumerate(lens):
        out[i, :l] = self.mask_index
      return out

  def _ddpm_caching_update(self, x, t, dt, p_x0=None, condition=None, conf=None, curr_embed=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      # p_x0 = self.forward(x, sigma_t, condition).exp()
      log_p_x0 = self.forward(x, sigma_t, condition, curr_embed=curr_embed)
      if self.config.sampling.nucleus_p < 1:
        p_x0 = log_p_x0.exp()
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
      else:
        p_x0 = log_p_x0.exp()
    
    move_chance_t = move_chance_t.to(torch.float64)
    move_chance_s = move_chance_s.to(torch.float64)
    p_x0 = p_x0.to(torch.float64)
    
    assert move_chance_t.ndim == p_x0.ndim
    if self.config.sampling.remdm_mode is None:
      q_xs = p_x0 * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      _x = _sample_categorical(q_xs)
      
      copy_flag = (x != self.mask_index).to(x.dtype)
      xs = copy_flag * x + (1 - copy_flag) * _x
    elif self.config.sampling.remdm_mode == "cap":
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma = min(self.config.sampling.eta, (1 - alpha_s) / alpha_t)
      else:
        sigma = self.config.sampling.eta
      q_xs = p_x0 * (1 - sigma)
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)

      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
    elif self.config.sampling.remdm_mode == "rescale":
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma_max = min(1, (1 - alpha_s) / alpha_t)
      else:
        sigma_max = 1
      sigma = self.config.sampling.eta * sigma_max
      q_xs = p_x0 * (1 - sigma)
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
    elif self.config.sampling.remdm_mode == "conf":
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma_max = min(1, (1 - alpha_s) / alpha_t)
      else:
        sigma_max = 1
      eta = conf.softmax(dim=-1)
      masked_flag = (x == self.mask_index).to(torch.bool)
      eta[masked_flag] = 0
      sigma = eta * sigma_max
      q_xs = p_x0 * (1 - sigma[:, :, None])
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma[:, :, None]) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
      # update conf
      unmask_mask = (x == self.mask_index) & (xs != self.mask_index)
      batch_indices = torch.arange(xs.shape[0])[:, None]
      feature_indices = torch.arange(xs.shape[1])
      conf_values = - p_x0[batch_indices, feature_indices, xs]
      conf[unmask_mask] = conf_values[unmask_mask]
      remask_mask = (x != self.mask_index) & (xs == self.mask_index)
      conf[remask_mask] = -torch.inf
    elif self.config.sampling.remdm_mode == "loop":
      time = t[0].item()
      # compute alpha_t and alpha_s
      if time > self.config.sampling.t_on:
        move_chance_t = (1 - (1 - t) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
        move_chance_s = (1 - (1 - t + dt) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
      elif time <= self.config.sampling.t_off:
        move_chance_t = (t * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
        move_chance_s = ((t - dt) * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
      else:
        move_chance_t, move_chance_s = None, None
      # use MDLM
      if time > self.config.sampling.t_on or time <= self.config.sampling.t_off:
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        xs = copy_flag * x + (1 - copy_flag) * _x
      else: # use ReMDM
        sigma = self.config.sampling.eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((self.config.sampling.alpha_on - (1 - sigma) * self.config.sampling.alpha_on) / (1 - self.config.sampling.alpha_on))
        q_xs_2[..., self.mask_index] = (1 - self.config.sampling.alpha_on - self.config.sampling.alpha_on * sigma) / (1 - self.config.sampling.alpha_on)
        copy_flag = (x != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q_xs)
    else:
      raise ValueError(f"Invalid remdm_mode: {self.config.sampling.remdm_mode}")

    return p_x0, xs, conf

  def _ddpm_update(self, x, t, dt, condition=None, curr_embed=None):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning, condition, curr_embed=curr_embed)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _remaskator_update(self, x, t, dt, condition=None, curr_embed=None):
    """Single-step update guided by Remaskator.

    Algorithm (per request):
      input: x_t, t, dt
      1) compute p_x0 using forward
      2) sample _x ~ p_x0 for masked positions, keep unmasked
      3) apply remaskator to _x to predict per-token mistake probs
      4) temperature-scale logits and sample positions via Gumbel-top-k (no softmax)
      5) sample without replacement N=(1-alpha_s)*L tokens, where s=t-dt, L=seq_len
      6) set those tokens back to mask to be regenerated in the next steps
    """
    # Step 1: compute p_x0 at time t
    sigma_t, _ = self.noise(t)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    log_p_x0 = self.forward(x, sigma_t, condition, curr_embed=curr_embed)
    if self.config.sampling.nucleus_p < 1:
      p_x0 = log_p_x0.exp()
      sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
      top_p_mask[..., 0] = True
      nucleus_probs = sorted_probs * top_p_mask
      nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
      p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
    else:
      p_x0 = log_p_x0.exp()

    # Step 2: sample proposal _x_0 ~ p_x0 keeping masked tokens
    _x = _sample_categorical(p_x0)
    copy_flag = (x != self.mask_index).to(x.dtype)
    x_prop = copy_flag * x + (1 - copy_flag) * _x

    if self.remaskator is not None:
      # Step 3: remaskator logits of being wrong for each token
      # Build conditioning embeddings for remaskator if available
      with torch.no_grad():
        remask_cond = condition
        if self.config.remaskator.global_conditioning:
          pos_logits = self.remaskator(x_prop, remask_cond)  # (batch, L)
        else:
          pos_logits = self.remaskator(x_prop)

        # save distribution of logits (to float32)
        # import matplotlib.pyplot as plt
        # logits_np = pos_logits.to(torch.float32).detach().cpu().numpy().flatten()
        # plt.hist(logits_np, bins=100)
        # plt.axvline(0, color='red', linestyle='--', linewidth=1)
        # ratio_more_than_zero = (pos_logits > 0).sum().item() / pos_logits.numel()
        # alpha = torch.exp(-sigma_t)
        # plt.title(f'Distribution of logits, error rate: {ratio_more_than_zero:.2f}, alpha: {alpha[0].item():.2f}')
        # plt.savefig('pos_logits_dist.png')
        # plt.close()
        # 1 / 0
      # Step 4: temperature-scale logits and avoid explicit softmax
      # Use Gumbel-Top-k on logits/T which is equivalent to sampling
      # without replacement from softmax(logits/T), but numerically stable
      temperature = max(1e-6, float(self.remaskator_temperature))
      scaled = pos_logits / temperature
    else:
      # all logits are random in case of no condition - no remasking
      batch, seq_len = x_prop.shape
      scaled = torch.randn(batch, seq_len, device=x_prop.device)
    
    # Step 5: sample N positions without replacement, N=(1-alpha_s)*L, s=t-dt
    # Compute alpha_s from sigma_s like in DDPM update: alpha_s = 1 - exp(-sigma_s)
    sigma_s, _ = self.noise(t - dt)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    alpha_s = 1 - torch.exp(-sigma_s)  # (batch,)
  
    L = x.shape[1]
    # number to remask per sample; clamp between 0 and L
    N_float = alpha_s * L
    N = torch.ceil(N_float).to(torch.int64)  # (batch,)
    N = torch.clamp(N, min=0, max=L) 

    # Gumbel-top-k sampling without replacement per batch row, using logits
    gumbel = -torch.log(-torch.log(torch.rand_like(scaled) + 1e-10) + 1e-10)
    scores = scaled + gumbel
    # argsort descending and take top-k per row
    topk_indices = torch.argsort(scores, dim=1, descending=True)

    # Build mask positions to re-mask per sample
    batch_size, seq_len = x.shape
    remask_positions = torch.zeros_like(x, dtype=torch.bool)
    for b in range(batch_size):
      k = int(N[b].item())
      if k > 0:
        idx = topk_indices[b, :k]
        remask_positions[b, idx] = True

    # Step 6: set chosen positions to mask index
    x_next = x_prop.clone()
    x_next[remask_positions] = self.mask_index
    return x_next

  def _sample_condition_from_validation_texts(self, batch_size):
    """Sample texts from validation set and compute embeddings and sequence lengths on-the-fly.
    Returns: (embeddings_or_none, seq_lens_tensor, reference_texts_list)
    """
    # Lazy-load validation dataset
    if self._valid_text_dataset is None:
      import dataloader as _mdl_dataloader  # local import to avoid import-time cycles
      _, valid_loader = _mdl_dataloader.get_dataloaders(
        self.config, self.tokenizer, skip_train=True, valid_seed=self.config.seed)
      self._valid_text_dataset = valid_loader.dataset

    dataset = self._valid_text_dataset
    total = len(dataset)
    if total == 0:
      # No data; fall back to null condition
      cond, seq = self._sample_condition_null(batch_size)
      return cond, seq, [None] * batch_size

    # Choose random indices
    indices = torch.randint(low=0, high=total, size=(batch_size,), device='cpu')

    # Gather texts and sequence lengths
    pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
    if pad_token_id is None:
      pad_token_id = getattr(self.config, 'pad_token_id', 0)

    texts = []
    lengths = []
    for idx in indices.tolist():
      sample = dataset[idx]
      input_ids = sample['input_ids'] if isinstance(sample, dict) else sample
      if isinstance(input_ids, torch.Tensor):
        ids_tensor = input_ids
      else:
        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
      text = self.tokenizer.decode(ids_tensor.tolist(), skip_special_tokens=True)
      texts.append(text)
      lengths.append((ids_tensor != pad_token_id).sum().item())

    sampled_seq_lens = torch.tensor(lengths, dtype=torch.int64, device=self.device)

    # Compute embeddings if enabled; otherwise return null conditioning
    if not self.config.text_embedder.use_text_embedder or self.text_embedder is None:
      return None, sampled_seq_lens, texts

    # Ensure embedder is on correct device
    try:
      self.text_embedder.model = self.text_embedder.model.to(self.device)
      self.text_embedder.device = self.device
    except Exception:
      pass

    with torch.no_grad():
      sampled_embeddings = self.text_embedder(texts)
      if not isinstance(sampled_embeddings, torch.Tensor):
        sampled_embeddings = torch.tensor(sampled_embeddings)
      sampled_embeddings = sampled_embeddings.to(self.device, dtype=torch.float32)

    return sampled_embeddings, sampled_seq_lens, texts

  def _sample_condition_from_train_texts(self, batch_size):
    """Sample texts from train set and compute embeddings and sequence lengths on-the-fly.
    Returns: (embeddings_or_none, seq_lens_tensor, reference_texts_list)
    """
    # Lazy-load train dataset
    if self._train_text_dataset is None:
      import dataloader as _mdl_dataloader  # local import to avoid import-time cycles
      train_loader, _ = _mdl_dataloader.get_dataloaders(
        self.config, self.tokenizer, skip_valid=True)
      self._train_text_dataset = train_loader.dataset

    dataset = self._train_text_dataset
    total = len(dataset)
    if total == 0:
      # No data; fall back to null condition
      cond, seq = self._sample_condition_null(batch_size)
      return cond, seq, [None] * batch_size

    # Choose random indices
    indices = torch.randint(low=0, high=total, size=(batch_size,), device='cpu')

    # Gather texts and sequence lengths
    pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
    if pad_token_id is None:
      pad_token_id = getattr(self.config, 'pad_token_id', 0)

    texts = []
    lengths = []
    for idx in indices.tolist():
      sample = dataset[idx]
      input_ids = sample['input_ids'] if isinstance(sample, dict) else sample
      if isinstance(input_ids, torch.Tensor):
        ids_tensor = input_ids
      else:
        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
      text = self.tokenizer.decode(ids_tensor.tolist(), skip_special_tokens=True)
      texts.append(text)
      lengths.append((ids_tensor != pad_token_id).sum().item())

    sampled_seq_lens = torch.tensor(lengths, dtype=torch.int64, device=self.device)

    # Compute embeddings if enabled; otherwise return null conditioning
    if not self.config.text_embedder.use_text_embedder or self.text_embedder is None:
      return None, sampled_seq_lens, texts

    # Ensure embedder is on correct device
    try:
      self.text_embedder.model = self.text_embedder.model.to(self.device)
      self.text_embedder.device = self.device
    except Exception:
      pass

    with torch.no_grad():
      sampled_embeddings = self.text_embedder(texts)
      if not isinstance(sampled_embeddings, torch.Tensor):
        sampled_embeddings = torch.tensor(sampled_embeddings)
      sampled_embeddings = sampled_embeddings.to(self.device, dtype=torch.float32)

    return sampled_embeddings, sampled_seq_lens, texts

  def _sample_condition_null(self, batch_size):
    """Return null conditioning (no embeddings) with default sequence lengths."""
    # Return None for embeddings (no conditioning)
    sampled_embeddings = None
    
    # Use default sequence length (model length) for all samples
    default_seq_len = self.config.model.length
    sampled_seq_lens = torch.full((batch_size,), default_seq_len, dtype=torch.int64, device=self.device)
    
    return sampled_embeddings, sampled_seq_lens

  def _sample_condition_from_gaussian_model(self, batch_size):
    """Sample condition embeddings (and optionally lengths) from Gaussian embedding diffusion model.

    Returns: (embeddings_tensor, seq_lens_tensor, reference_texts_list)
    """
    import os as _os
    from models.embedding_diffusion_module import EmbeddingDiffusionModule as _EmbeddingDiffusionModule

    ckpt_path = getattr(self.config.sampling, 'gaussian_checkpoint_path', '')
    if not ckpt_path or not _os.path.exists(ckpt_path):
      raise ValueError(
        f"Invalid or missing sampling.gaussian_checkpoint_path: '{ckpt_path}'. "
        "Provide a valid path to an EmbeddingDiffusionModule checkpoint.")

    # Lazy-load and cache the gaussian embedding model
    if not hasattr(self, '_gaussian_embedding_module') or self._gaussian_embedding_module is None:
      try:
        module = _EmbeddingDiffusionModule.load_from_checkpoint(ckpt_path, map_location=self.device)
      except Exception as e:
        raise RuntimeError(f"Failed to load EmbeddingDiffusionModule from '{ckpt_path}': {e}")
      module = module.to(self.device)
      module.eval()
      self._gaussian_embedding_module = module

    with torch.no_grad():
      embs, lengths = self._gaussian_embedding_module.sample(num_samples=batch_size, device=self.device)
      if not isinstance(embs, torch.Tensor):
        embs = torch.tensor(embs)
      embs = embs.to(self.device, dtype=torch.float32)
      if lengths is None:
        seq_lens = torch.full((batch_size,), int(self.config.model.length), dtype=torch.int64, device=self.device)
      else:
        if not isinstance(lengths, torch.Tensor):
          lengths = torch.tensor(lengths)
        seq_lens = lengths.to(self.device, dtype=torch.int64)

    reference_texts = [None] * batch_size
    return embs, seq_lens, reference_texts

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
      
    if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      num_steps = 1

    # sample condition based on config
    if self.sample_embeddings_from == 'train':
      condition, seq_lens, reference_texts = self._sample_condition_from_train_texts(batch_size_per_gpu)
    elif self.sample_embeddings_from == 'validation':
      condition, seq_lens, reference_texts = self._sample_condition_from_validation_texts(batch_size_per_gpu)
    elif self.sample_embeddings_from == 'gaussian':
      condition, seq_lens, reference_texts = self._sample_condition_from_gaussian_model(batch_size_per_gpu)
    elif self.sample_embeddings_from is None:
      condition, seq_lens = self._sample_condition_null(batch_size_per_gpu)
      reference_texts = [None] * batch_size_per_gpu
    else:
      raise ValueError(f"Invalid sample_embeddings_from: {self.sample_embeddings_from}. Must be 'validation', 'train', 'gaussian', or None.")

    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length,
      lens=seq_lens).to(self.device)

    # Optional: setup per-step trajectory recording
    save_traj = False
    max_to_save = 0
    try:
      save_traj = bool(self.config.eval.save_sampling_trajectory)
      max_to_save = int(self.config.eval.max_trajectories_to_save)
    except Exception:
      save_traj = False
      max_to_save = 0
    is_main_process = True
    try:
      is_main_process = (self.trainer.global_rank == 0)
    except Exception:
      pass
    remaining_to_save = max(0, max_to_save - getattr(self, '_trajectories_saved', 0))
    indices_to_save = list(range(min(remaining_to_save, x.shape[0])) if (save_traj and is_main_process and remaining_to_save > 0) else [])
    traj_records = {idx: [] for idx in indices_to_save}
    def _record_state(current_x, current_t):
      if len(indices_to_save) == 0:
        return
      if isinstance(current_t, torch.Tensor):
        t_scalar = float(current_t.view(-1)[0].item())
      else:
        t_scalar = float(current_t)
      for _idx in indices_to_save:
        _xi = current_x[_idx].detach().to('cpu')
        tokens_list = _xi.to(torch.int64).tolist()
        # Robust detokenization including special tokens; insert [MASK] for our custom mask id
        tokens_str = []
        for _id in tokens_list:
          if _id == self.mask_index:
            tokens_str.append('[MASK]')
          else:
            try:
              tok = self.tokenizer.convert_ids_to_tokens([_id])[0]
            except Exception:
              tok = self.tokenizer.unk_token if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None else '<unk>'
            tokens_str.append(tok)
        try:
          x_text = self.tokenizer.convert_tokens_to_string(tokens_str)
        except Exception:
          x_text = ' '.join(tokens_str)
        traj_records[_idx].append({'t': f"{t_scalar:.6f}", 'x': x_text, 'x_tokens': tokens_list})
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    # Determine the last step index (exclusive) where conditioning is applied
    cond_until_ratio = self.config.text_embedder.use_condition_during_sampling_until
    cond_until_ratio = max(0.0, min(1.0, float(cond_until_ratio)))
    cond_until_step = int(round(cond_until_ratio * num_steps))

    # Plan EMA-based condition embedding updates across the conditioned steps
    ema_decay = float(self.config.text_embedder.embedding_ema_decay)
    num_cond_updates = int(self.config.text_embedder.num_embedding_updates)
    update_steps = set()
    if (
      num_cond_updates > 0
      and cond_until_step > 0
      and (condition is not None)
      and (self.text_embedder is not None)
    ):
      # Space updates roughly evenly within [0, cond_until_step - 1]
      for j in range(num_cond_updates):
        step_idx = int((j + 1) * cond_until_step / (num_cond_updates + 1))
        step_idx = min(max(step_idx, 0), max(cond_until_step - 1, 0))
        update_steps.add(step_idx)

    confident_score = - torch.ones_like(x, device=self.device).to(torch.float64) * torch.inf
    
    # Initialize curr_embed for sub conditioning
    # curr_embed = None
    # if True:
    #   curr_embed = self.indices_to_text_embeddings(x)
    
    # Variables to store first step metrics
    first_step_accuracy = None
    first_step_levenshtein = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      
      if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
        t = torch.ones(x.shape[0], 1, device=self.device) * 0.999999
        
      # Use condition only while i < cond_until_step
      step_condition = condition if (condition is not None and i < cond_until_step) else None

      # Update curr_embed for sub conditioning at each step
      # For original_vae_decoder, curr_embed is not used (decoder ignores it)
      if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
        curr_embed = None
      elif True:
        curr_embed = self.indices_to_text_embeddings(x)

      # Compute first step metrics (only on first iteration and if we have reference texts)
      if i == 0 and reference_texts is not None and any(ref is not None for ref in reference_texts):
        # Get predictions for first step
        sigma_t, _ = self.noise(t)
        if sigma_t.ndim > 1:
          sigma_t = sigma_t.squeeze(-1)
        
        with torch.no_grad():
          log_p_x0 = self.forward(x, sigma_t, step_condition, curr_embed=curr_embed)
          predicted_x0 = log_p_x0.argmax(dim=-1)
          
          # Convert reference texts to tokens for comparison
          reference_token_lists = []
          for ref_text in reference_texts:
            if ref_text is not None:
              ref_tokens = self.tokenizer.encode(ref_text)
              # Pad or truncate to match sequence length
              if len(ref_tokens) < x.shape[1]:
                ref_tokens.extend([self.tokenizer.pad_token_id] * (x.shape[1] - len(ref_tokens)))
              else:
                ref_tokens = ref_tokens[:x.shape[1]]
              reference_token_lists.append(ref_tokens)
            else:
              reference_token_lists.append([self.tokenizer.pad_token_id] * x.shape[1])
          
          if reference_token_lists:
            reference_tokens = torch.tensor(reference_token_lists, device=x.device, dtype=torch.long)
            first_step_accuracy, first_step_levenshtein = _compute_first_step_accuracy_and_levenshtein(
              self.tokenizer, predicted_x0, reference_tokens)
        
      if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
        with torch.no_grad():
          log_p_x0 = self.forward(x, None, step_condition, curr_embed=curr_embed)
          x = _sample_categorical(log_p_x0.exp())
          break

      # Optionally update the condition embedding via EMA using a sampled x0
      if (i in update_steps) and (step_condition is not None):
        sigma_t, _ = self.noise(t)
        if sigma_t.ndim > 1:
          sigma_t = sigma_t.squeeze(-1)
          
        # log_p_x0 = self.forward(x, sigma_t, step_condition, curr_embed=curr_embed)
        # probs_x0 = log_p_x0.exp()
        # x0_sample = _sample_categorical(probs_x0)

        step_condition = condition.detach()
        p_x0_cache = None
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt, step_condition, curr_embed=curr_embed)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next, confident_score = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache, condition=step_condition, conf=confident_score, curr_embed=curr_embed)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      elif self.sampler == 'remaskator':
        # Disable caching when using remaskator
        p_x0_cache = None
        # Gate remaskator usage by configured interval [t_off, t_on]
        # t is in [1, eps] decreasing. Convert to scalar per-batch t for gating.
        t_scalar = float(t.view(-1)[0].item())
        if (t_scalar >= self.remaskator_t_off) and (t_scalar <= self.remaskator_t_on):
          x = self._remaskator_update(x, t, dt, step_condition)
        else:
          # Fallback to plain DDPM step outside the interval
          x = self._ddpm_update(x, t, dt, step_condition)
      else:
        x = self._analytic_update(x, t, dt, step_condition)

      # Record state after update at this timestep t
      _record_state(x, t)

    if self.config.sampling.noise_removal and not self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      final_condition = condition if (condition is not None and num_steps <= cond_until_step) else None
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t, final_condition)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning, final_condition).argmax(dim=-1)
      # Note: per request, do not record post noise-removal state

    # Persist recorded trajectories to disk (rank 0 only)
    if len(indices_to_save) > 0 and is_main_process:
      traj_dir = 'sampling_trajectories'
      os.makedirs(traj_dir, exist_ok=True)
      base_id = getattr(self, '_trajectories_saved', 0)
      for j, _idx in enumerate(indices_to_save):
        traj_id = base_id + j
        # Write a JSONL with one record per step: {'t', 'x_tokens', 'x'}
        jsonl_path = os.path.join(traj_dir, f'traj_{traj_id}.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
          for rec in traj_records[_idx]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
      # Update global counter
      self._trajectories_saved = base_id + len(indices_to_save)
    return x, reference_texts, first_step_accuracy, first_step_levenshtein

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model.
    Returns: (samples_tensor, reference_texts_list, first_step_accuracy, first_step_levenshtein)
    """
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    self.backbone.to(torch.float32)
    self.noise.to(torch.float32)
    samples, reference_texts, first_step_accuracy, first_step_levenshtein = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples, reference_texts, first_step_accuracy, first_step_levenshtein

  def get_score(self, x, sigma, condition=None):
    if self.config.sub_conditioning.enabled:
      curr_embed = self.indices_to_text_embeddings(x)
    else:
      curr_embed = None
    
    model_output = self.forward(x, sigma, condition, curr_embed=curr_embed)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size, condition=None):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma, condition)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t, condition=None):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma, condition)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.change_time_scheduler:
      # Randomly select half of the indices to set to 0.999
      perm = torch.randperm(n, device=device)
      half_n = n // 2
      idx = perm[:half_n]
      _eps_t[idx] = 0.999
    
    if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      _eps_t = torch.ones(n, device=device) * 0.999999
    
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]

    # condition can be extracted from x0
    condition = self.indices_to_text_embeddings(x0)

    model_output_t0 = self.forward(x0, unet_conditioning, condition)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0):
    # condition can be extracted from x0
    condition = self.indices_to_text_embeddings(x0)

    # Convert x0 tensor to a list of integers and dump it to a file
    # x0_list = x0.cpu().numpy().tolist()
    # with open("x0_dump.txt", "w") as f:
    #     f.write(str(x0_list))
    # 1 / 0

    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance)
    
    # For original_vae_decoder, curr_embed is not used (decoder ignores it)
    # and computing embeddings from noisy/masked tokens is not meaningful
    if self.config.TYPE_OF_CONDITIONING == 'original_vae_decoder':
      curr_embed = None
    elif True:
      curr_embed = self.indices_to_text_embeddings(xt)
    else:
      curr_embed = None
    
    model_output = self.forward(xt, unet_conditioning, condition, curr_embed=curr_embed)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  # @torch.no_grad
  # def sample_subs_guidance(
  #   self, n_samples, stride_length, num_strides, dt=0.001):
  #   ones = torch.ones(n_samples, dtype=self.dtype,
  #                     device=self.device)

  #   num_steps = int(1 / dt)
  #   sampling_steps = 0
  #   intermediate_tokens = []
  #   target = None
  #   for _ in range(num_strides + 1):
  #     p_x0_cache = None
  #     x = self._sample_prior(
  #       n_samples,
  #       self.config.model.length).to(self.device)
  #     if target is not None:
  #       x[:, : -stride_length] = target
  #     for i in range(num_steps + 1):
  #       p_x0_cache, x_next = self._ddpm_caching_update(
  #         x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
  #       if (not torch.allclose(x_next, x)
  #           or self.time_conditioning):
  #         p_x0_cache = None
  #         sampling_steps += 1
  #       x = x_next
  #     x = self.forward(x, 0 * ones).argmax(dim=-1)
  #     intermediate_tokens.append(
  #       x[:, :stride_length].cpu().numpy())
  #     target = x[:, stride_length:]
    
  #   intermediate_tokens.append(target.cpu().numpy())
  #   intermediate_text_samples = []
  #   sequence_lengths = ((
  #     np.concatenate(intermediate_tokens, axis=1)[:, 1:]
  #     == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
  #   for i in range(2, len(intermediate_tokens) + 1):
  #     intermediate_text_samples.append(
  #       self.tokenizer.batch_decode(
  #         np.concatenate(intermediate_tokens[:i], axis=1)))
  #   return (sampling_steps, intermediate_text_samples,
  #           sequence_lengths)

  # def restore_model_and_semi_ar_sample(
  #     self, stride_length, num_strides, dt=0.001):
  #   """Generate samples from the model."""
  #   # Lightning auto-casting is not working in this method for some reason
  #   if self.ema:
  #     self.ema.store(itertools.chain(
  #       self.backbone.parameters(),
  #       self.noise.parameters()))
  #     self.ema.copy_to(itertools.chain(
  #       self.backbone.parameters(),
  #       self.noise.parameters()))
  #   self.backbone.eval()
  #   self.noise.eval()
  #   (sampling_steps, samples,
  #    sequence_lengths) = self.sample_subs_guidance(
  #     n_samples=self.config.loader.eval_batch_size,
  #     stride_length=stride_length,
  #     num_strides=num_strides, 
  #     dt=dt)
  #   if self.ema:
  #     self.ema.restore(itertools.chain(
  #       self.backbone.parameters(),
  #       self.noise.parameters()))
  #   self.backbone.train()
  #   self.noise.train()
  #   return sampling_steps, samples, sequence_lengths

  def indices_to_text_embeddings(self, indices, attention_mask=None):
    """Convert batch of token indices to text embeddings.
    
    Args:
        indices: torch.Tensor of shape (batch_size, sequence_length)
                containing token indices
        attention_mask: torch.Tensor of shape (batch_size, sequence_length)
    Returns:
        torch.Tensor: Text embeddings from the text embedder
    """
    import os

    if self.text_embedder is None:
      return None

    # Convert indices to text
    text_samples = self.tokenizer.batch_decode(indices, skip_special_tokens=True)

    # Logging: write the decoded texts to a file
    # log_dir = "./embedding_logs"
    # os.makedirs(log_dir, exist_ok=True)
    # text_log_path = os.path.join(log_dir, "texts.log")
    # with open(text_log_path, "a", encoding="utf-8") as f:
    #   for text, tokenized_text in zip(text_samples, indices):
    #     f.write(text.replace("\n", "\\n") + "\n")
    #     f.write(str(tokenized_text.tolist()) + "\n")
    #     f.write("=" * 100 + "\n")

    # Get text embeddings
    text_embeddings = self.text_embedder(text_samples)

    # During training, optionally add Gaussian noise to condition embeddings
    if (isinstance(text_embeddings, torch.Tensor)
        and self.config.text_embedder.noise > 0.0
        and self.training):
      noise_std = float(self.config.text_embedder.noise)
      text_embeddings = text_embeddings + noise_std * torch.randn_like(text_embeddings)

    # Logging: write the embeddings to a file
    # emb_log_path = os.path.join(log_dir, "embeddings.log")
    # with open(emb_log_path, "a", encoding="utf-8") as f:
    #   if isinstance(text_embeddings, torch.Tensor):
    #     emb_np = text_embeddings.detach().cpu().to(torch.float32).numpy()
    #     for emb in emb_np:
    #       f.write(" ".join([f"{x:.6f}" for x in emb.flatten()]) + "\n")
    #   else:
    #     # If not a tensor, just str() it
    #     f.write(str(text_embeddings) + "\n")      
    #   f.write("=" * 100 + "\n")

    # Ensure embeddings are on the same device as indices (safety check)
    if isinstance(text_embeddings, torch.Tensor):
      text_embeddings = text_embeddings.to(indices.device)

    return text_embeddings