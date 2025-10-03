import os
import json

# Set tokenizer parallelism to false to avoid warnings in multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
from tqdm import tqdm
import numpy as np
import random
from collections import defaultdict

import dataloader
import diffusion
import utils
from safetensors.torch import load_file
from lightning_json_logger import LightningJSONLogger

import mauve

import setproctitle
setproctitle.setproctitle("python main.py")



omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _lazy_get_spacy_tokenizer():
  """Return a spaCy tokenizer if available, otherwise None.

  Loaded lazily to avoid startup overhead and hard dependency on spaCy model.
  """
  try:
    import spacy  # type: ignore
    # Load small English model; if unavailable, this will raise and we'll fallback
    nlp = spacy.load("en_core_web_sm")
    return nlp.tokenizer
  except Exception:
    return None


def _iter_ngrams(tokens, n):
  """Yield n-grams from a token sequence without external deps."""
  if n <= 0:
    return
  length = len(tokens)
  if length < n:
    return
  for idx in range(length - n + 1):
    yield tuple(tokens[idx: idx + n])


def compute_diversity(all_texts_list):
  """Compute n-gram repetition and aggregate diversity metrics for generated texts.

  Returns a dict with keys like '2gram_repetition', '3gram_repetition', '4gram_repetition',
  and 'diversity' which aggregates across n-gram levels.
  """
  ngram_range = (2, 3, 4)

  tokenizer = _lazy_get_spacy_tokenizer()
  token_lists = []
  for sentence in all_texts_list:
    if tokenizer is not None:
      # Use spaCy tokenizer if available
      tokens = [str(token) for token in tokenizer(sentence)]
    else:
      # Fallback: simple whitespace tokenization
      tokens = sentence.split()
    token_lists.append(tokens)

  ngram_unique_sets = {n: set() for n in ngram_range}
  ngram_total_counts = defaultdict(int)

  metrics = {}
  for n in ngram_range:
    for tokens in token_lists:
      ngrams_for_tokens = list(_iter_ngrams(tokens, n))
      if not ngrams_for_tokens:
        continue
      ngram_unique_sets[n].update(ngrams_for_tokens)
      ngram_total_counts[n] += len(ngrams_for_tokens)

    total = ngram_total_counts[n]
    unique = len(ngram_unique_sets[n])
    if total == 0:
      repetition = 0.0
    else:
      repetition = 1.0 - (unique / float(total))
    metrics[f"{n}gram_repetition"] = repetition

  diversity_product = 1.0
  for n in ngram_range:
    repetition = metrics.get(f"{n}gram_repetition", 0.0)
    diversity_component = 1.0 - repetition
    diversity_product *= diversity_component
  metrics["diversity"] = diversity_product

  return metrics

def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config, strict=False)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)
    
    
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


def _compute_accuracy_and_levenshtein(tokenizer, generated_texts, reference_texts):
  """Compute mean token-level accuracy and Levenshtein distance between generated and reference texts.
  
  Args:
    tokenizer: The tokenizer used to encode texts
    generated_texts: List of generated text strings
    reference_texts: List of reference text strings (can contain None values)
    
  Returns:
    tuple: (mean_accuracy, mean_levenshtein_distance) across all valid pairs, or (0.0, 0.0) if no valid pairs
  """
  if not generated_texts or not reference_texts:
    return 0.0, 0.0
    
  total_accuracy = 0.0
  total_levenshtein = 0.0
  valid_pairs = 0
  
  for gen_text, ref_text in zip(generated_texts, reference_texts):
    # Skip if reference text is None (unconditional generation)
    if ref_text is None or ref_text == '':
      continue
      
    # Tokenize both texts
    gen_tokens = tokenizer.encode(gen_text)
    ref_tokens = tokenizer.encode(ref_text)
    
    # Skip if reference is empty
    if len(ref_tokens) == 0:
      continue
      
    # Compute token-level accuracy for this pair
    min_length = min(len(gen_tokens), len(ref_tokens))
    if min_length > 0:
      # Count matching tokens (prefix accuracy)
      matches = sum(1 for i in range(min_length) if gen_tokens[i] == ref_tokens[i])
      pair_accuracy = matches / len(ref_tokens)  # Normalize by reference length
    else:
      pair_accuracy = 0.0
    
    # Compute Levenshtein distance
    levenshtein_dist = _compute_levenshtein_distance(gen_tokens, ref_tokens)
    # Normalize by reference length to get a relative distance
    normalized_levenshtein = levenshtein_dist / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    total_accuracy += pair_accuracy
    total_levenshtein += normalized_levenshtein
    valid_pairs += 1
  
  # Return mean accuracy and mean normalized Levenshtein distance across all valid pairs
  if valid_pairs > 0:
    return total_accuracy / valid_pairs, total_levenshtein / valid_pairs
  else:
    return 0.0, 0.0


def _get_validation_texts(config, tokenizer, max_samples, seed=42):
  """Get validation texts, subsampled to match generated samples count."""
  # Fix seed for reproducibility
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  
  # Get validation dataloader
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=seed)
  
  # Collect validation texts
  validation_texts = []
  for batch in tqdm(valid_ds, desc="Collecting validation texts"):
    texts = tokenizer.batch_decode(batch['input_ids']) #, skip_special_tokens=True)
    validation_texts.extend(texts)
    # Stop early if we have enough samples for efficiency
    if len(validation_texts) >= max_samples * 2:  # Collect more to subsample from
      break
  
  # Subsample to max_samples
  if len(validation_texts) > max_samples:
    validation_texts = np.random.choice(validation_texts, size=max_samples, replace=False).tolist()
  
  return validation_texts


def _compute_mauve(generated_texts, reference_texts):
  """Compute MAUVE metric."""
  if mauve is None:
    print("MAUVE not available. Skipping MAUVE computation.")
    return None
  
  result = mauve.compute_mauve(
    p_text=reference_texts,
    q_text=generated_texts,
    device_id=0 if torch.cuda.is_available() else -1,
    verbose=False
  )
  return result.mauve


def compute_number_of_parameters(module):
  return sum(p.numel() for p in module.parameters())

def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  logger.info(f'Number of parameters: {compute_number_of_parameters(model.backbone):,}')
  model.gen_ppl_metric.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  stride_length = config.sampling.stride_length
  num_strides = config.sampling.num_strides
  all_text_samples = []
  all_reference_texts = []
  all_first_step_accuracies = []
  all_first_step_levenshteins = []
  
  for _ in tqdm(range(config.sampling.num_sample_batches)):
    if config.sampling.semi_ar:
      _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
        stride_length=stride_length,
        num_strides=num_strides,
        dt=1 / config.sampling.steps)
      text_samples = intermediate_samples[-1]
      all_text_samples.extend(text_samples)
      # Note: Samples generated using semi-ar method
      # need to to be processed before computing generative perplexity
      # since these samples contain numerous <|endoftext|> tokens
      # and diffusion.compute_generative_perplexity() discards
      # any text after the first EOS token.
    else:
      samples, reference_texts, first_step_acc, first_step_lev = model.restore_model_and_sample(
        num_steps=config.sampling.steps)
      text_samples = model.tokenizer.batch_decode(samples) #, skip_special_tokens=True)
      all_text_samples.extend(text_samples)
      all_reference_texts.extend(reference_texts)
      # Collect first step metrics if available
      if first_step_acc is not None:
        all_first_step_accuracies.append(first_step_acc)
      if first_step_lev is not None:
        all_first_step_levenshteins.append(first_step_lev)
      model.compute_generative_perplexity(text_samples)
  
  # compute accuracy and Levenshtein distance
  accuracy, levenshtein_distance = _compute_accuracy_and_levenshtein(tokenizer, all_text_samples, all_reference_texts)
  
  # Get validation texts matching the number of generated samples
  validation_texts = _get_validation_texts(config, tokenizer, max_samples=len(all_text_samples))
  logger.info(f'Collected {len(validation_texts)} validation texts.')
  
  # dump generated and validation/reference texts to file
  import os

  # Create directories for generated and validation samples
  generated_dir = 'generated_samples'
  validation_dir = 'validation_samples'
  reference_dir = 'reference_texts'
  paired_dir = 'paired_samples'
  os.makedirs(generated_dir, exist_ok=True)
  os.makedirs(validation_dir, exist_ok=True)
  os.makedirs(reference_dir, exist_ok=True)
  os.makedirs(paired_dir, exist_ok=True)

  # Write each generated sample to a separate file
  for idx, text in enumerate(all_text_samples):
    with open(os.path.join(generated_dir, f'sample_{idx}.txt'), 'w') as f:
      f.write(text)

  # Write each validation sample to a separate file
  for idx, text in enumerate(validation_texts):
    with open(os.path.join(validation_dir, f'sample_{idx}.txt'), 'w') as f:
      f.write(text)

  # Write each reference text (used for conditioning) to a separate file
  if all_reference_texts:
    for idx, text in enumerate(all_reference_texts):
      # Some entries can be None if null conditioning was used
      safe_text = '' if text is None else text
      with open(os.path.join(reference_dir, f'sample_{idx}.txt'), 'w') as f:
        f.write(safe_text)

  # Write each generated/reference pair to a single file
  if all_reference_texts:
    for idx, gen_text in enumerate(all_text_samples):
      ref_text = all_reference_texts[idx] if idx < len(all_reference_texts) else ''
      if ref_text is None:
        ref_text = ''
      content = f"sample:\n\n{gen_text}\n\n=== reference\n\n{ref_text}"
      with open(os.path.join(paired_dir, f'pair_{idx}.txt'), 'w') as f:
        f.write(content)

  # Compute and print generative perplexity
  if not config.sampling.semi_ar:
    gen_ppl = model.gen_ppl_metric.compute().item()
    print(f'Generative perplexity: {gen_ppl:.2f}')
  
  # Compute MAUVE metric
  logger.info('Computing MAUVE metric...')
  mauve_score = _compute_mauve(
    generated_texts=all_text_samples,
    reference_texts=validation_texts
  )
  
  if mauve_score is not None:
    print(f'MAUVE score: {mauve_score * 100:.4f}')

  # Compute diversity metrics for generated texts
  logger.info('Computing diversity metrics...')
  diversity_metrics = compute_diversity(all_text_samples)
  # Pretty print metrics
  print('Diversity metrics:')
  for key in sorted(diversity_metrics.keys()):
    value = diversity_metrics[key]
    if isinstance(value, float):
      print(f'  {key}: {value * 100:.6f}')
    else:
      print(f'  {key}: {value * 100}')

  # Compute diversity metrics for validation texts
  logger.info('Computing diversity metrics for validation texts...')
  validation_diversity_metrics = compute_diversity(validation_texts)
  print('Validation diversity metrics:')
  for key in sorted(validation_diversity_metrics.keys()):
    value = validation_diversity_metrics[key]
    if isinstance(value, float):
      print(f'  {key}: {value * 100:.6f}')
    else:
      print(f'  {key}: {value * 100}')
  
  # Print accuracy and Levenshtein distance metrics
  print(f'Mean token accuracy: {accuracy * 100:.4f}%')
  print(f'Mean normalized Levenshtein distance: {levenshtein_distance:.4f}')
  
  # Compute and print first step metrics
  first_step_accuracy = None
  first_step_levenshtein = None
  if all_first_step_accuracies:
    first_step_accuracy = sum(all_first_step_accuracies) / len(all_first_step_accuracies)
    print(f'First step accuracy: {first_step_accuracy * 100:.4f}%')
  if all_first_step_levenshteins:
    first_step_levenshtein = sum(all_first_step_levenshteins) / len(all_first_step_levenshteins)
    print(f'First step normalized Levenshtein distance: {first_step_levenshtein:.4f}')
  
  # return 7 metrics: mauve, diversity, perplexity, accuracy, levenshtein_distance, first_step_accuracy, first_step_levenshtein
  return mauve_score, diversity_metrics['diversity'], gen_ppl, accuracy, levenshtein_distance, first_step_accuracy, first_step_levenshtein

def test_condition_embedding_dependence(config, logger, tokenizer, num_test_batches=5, std_dev=1.0):
  """
  Simple test: load model, use fixed input with zero vs random embeddings, 
  compute divergence between backbone outputs.
  """
  logger.info('Testing condition embedding dependence (simple version).')
  
  model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  
  model.backbone.eval()
  
  batch_size = config.loader.eval_batch_size
  cond_dim_embedding = model.text_embedder.cond_dim if model.text_embedder is not None else config.model.cond_dim_embedding
  
  # Get some fixed inputs from validation data
  _, valid_ds = dataloader.get_dataloaders(config, tokenizer, skip_train=True, valid_seed=42)
  batch = next(iter(valid_ds))
  fixed_input = batch['input_ids'][:batch_size].to(model.device)
  
  # Fixed sigma (timestep)
  fixed_sigma = torch.zeros(batch_size, device=model.device)
  
  with torch.no_grad():
    # Zero embedding condition
    zero_condition = torch.zeros(batch_size, cond_dim_embedding, device=model.device)
    
    # Random normal embedding condition  
    random_condition = torch.randn(batch_size, cond_dim_embedding, device=model.device)
    
    # Forward pass with zero condition
    logits_zero = model.backbone(fixed_input, fixed_sigma, zero_condition)
    
    # Forward pass with random condition  
    logits_random = model.backbone(fixed_input, fixed_sigma, random_condition)
    
    # Convert to probabilities
    probs_zero = torch.softmax(logits_zero, dim=-1)
    probs_random = torch.softmax(logits_random, dim=-1)
    
    # Compute KL divergence: KL(random || zero)
    kl_div = torch.nn.functional.kl_div(
      torch.log(probs_zero + 1e-8), 
      probs_random, 
      reduction='batchmean'
    )
    
    # Compute Jensen-Shannon divergence
    m = (probs_zero + probs_random) / 2
    js_div = 0.5 * torch.nn.functional.kl_div(torch.log(probs_zero + 1e-8), m, reduction='batchmean') + \
             0.5 * torch.nn.functional.kl_div(torch.log(probs_random + 1e-8), m, reduction='batchmean')

    mse = torch.nn.functional.mse_loss(logits_zero, logits_random)
  
  model.backbone.train()
  
  print(f"\n{'='*50}")
  print("CONDITION EMBEDDING DEPENDENCE TEST")
  print(f"{'='*50}")
  print(f"KL Divergence (random || zero): {kl_div.item():.6f}")
  print(f"JS Divergence: {js_div.item():.6f}")
  print(f"MSE: {mse.item():.12f}")
  print(f"{'='*50}")


def _ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')

  model = _load_from_checkpoint(config=config,
                                tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=False)  # Disable default logger for evaluation
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  
  # Create JSON logger as primary logger - use hydra.run.dir as save directory
  experiment_name = config.get('experiment_name', 'mdlm_training')
  # Use hydra's current working directory (which is hydra.run.dir due to chdir: true)
  save_dir = os.getcwd()
  json_logger = LightningJSONLogger(
    save_dir=save_dir,
    experiment_name=experiment_name,
    version=config.get('version', '0'),
    update_freq=config.get('plot_update_freq', 10)  # Update plots every 10 steps by default (less frequent)
  )
  
  # Log configuration
  json_logger.log_config(omegaconf.OmegaConf.to_object(config))

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds = dataloader.get_dataloaders(
    config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  model = diffusion.Diffusion(
    config, tokenizer=valid_ds.tokenizer)

  # Setup logger - use only JSON logger
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=json_logger)

  state_dict = None
  if ckpt_path:
    if ckpt_path.endswith('ckpt'):
      print(f"Loading checkpoint from {ckpt_path}")
      state_dict = torch.load(ckpt_path)["state_dict"]
    elif ckpt_path.endswith('safetensors'):
      print(f"Loading safetensors from {ckpt_path}")
      state_dict = load_file(ckpt_path)
    else:
      raise ValueError(f"Unknown checkpoint format for {ckpt_path}")
    model.load_state_dict(state_dict, strict=False)
  else:
    print('Training from scratch')
  
  try:
    trainer.fit(model, train_ds, valid_ds)
  finally:
    # Ensure plots are finalized even if training is interrupted
    json_logger.finalize()


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  if config.mode == 'sample_eval':
    mauve_score, diversity, gen_ppl, accuracy, levenshtein_distance, first_step_accuracy, first_step_levenshtein = generate_samples(config, logger, tokenizer)
    logger.info(f"MAUVE score: {mauve_score * 100:.4f}")
    logger.info(f"Diversity: {diversity * 100:.4f}")
    logger.info(f"Generative perplexity: {gen_ppl:.2f}")
    logger.info(f"Mean token accuracy: {accuracy * 100:.4f}%")
    logger.info(f"Mean normalized Levenshtein distance: {levenshtein_distance:.4f}")
    if first_step_accuracy is not None:
      logger.info(f"First step accuracy: {first_step_accuracy * 100:.4f}%")
    if first_step_levenshtein is not None:
      logger.info(f"First step normalized Levenshtein distance: {first_step_levenshtein:.4f}")

    # Append metrics to a JSONL file. Prefer eval.metrics_file if provided.
    metrics_file = None
    if 'eval' in config and isinstance(config.eval, omegaconf.DictConfig):
      metrics_file = config.eval.get('metrics_file', None)
    if metrics_file is None or metrics_file == '':
      metrics_file = os.path.join(os.path.dirname(__file__), 'metrics.jsonl')

    metrics_dir = os.path.dirname(metrics_file) or '.'
    os.makedirs(metrics_dir, exist_ok=True)
    if not os.path.exists(metrics_file):
      with open(metrics_file, 'w') as f:
        pass  # create the file if it doesn't exist

    with open(metrics_file, 'a') as f:
      metrics_dict = {
        'mauve': mauve_score,
        'diversity': diversity,
        'gen_ppl': gen_ppl,
        'accuracy': accuracy,
        'levenshtein_distance': levenshtein_distance,
        'seed': config.seed,
        'checkpoint': config.eval.checkpoint_path
      }
      if first_step_accuracy is not None:
        metrics_dict['first_step_accuracy'] = first_step_accuracy
      if first_step_levenshtein is not None:
        metrics_dict['first_step_levenshtein'] = first_step_levenshtein
      f.write(json.dumps(metrics_dict) + '\n')
  elif config.mode == 'ppl_eval':
    _ppl_eval(config, logger, tokenizer)
  elif config.mode == 'condition_dependence_test':
    test_condition_embedding_dependence(config, logger, tokenizer)
  else:
    _train(config, logger, tokenizer)


if __name__ == '__main__':
  main()