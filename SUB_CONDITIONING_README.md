# Sub Conditioning Configuration

This document describes the clean implementation of sub conditioning in the MDLM codebase.

## Overview

Sub conditioning is a technique that uses current state embeddings alongside reference condition embeddings during the diffusion process. It allows the model to dynamically adjust its conditioning based on the current state of the generation.

## Configuration Structure

Sub conditioning is now configured through a dedicated configuration section with three presets:

### Configuration Presets

1. **`sub_conditioning=disabled`** (default)
   - Disables all sub conditioning features
   - Uses standard conditioning only

2. **`sub_conditioning=basic`** 
   - Enables residual modulation only
   - Uses `(condition - curr_embed)` as additional conditioning

3. **`sub_conditioning=enabled`**
   - Enables full sub conditioning with all features
   - Includes residual modulation and learnable weighted combinations

### Configuration Options

```yaml
sub_conditioning:
  enabled: true/false                    # Enable sub conditioning
  use_residual_modulation: true/false    # Use (condition - curr_embed) residual
  use_weighted_sum: true/false          # Use learnable weights for combinations
```

## Usage

### Training with Sub Conditioning

```bash
# Use the clean training script
./my_scripts/train_sub_conditioning.sh

# Or specify preset directly
python main.py sub_conditioning=enabled [other options...]
```

### Sampling/Evaluation

```bash
# Sample with sub conditioning
./my_scripts/sample_sub_conditioning.sh

# Test conditioning dependence
./my_scripts/test_sub_conditioning.sh
```

### Custom Configuration

You can override specific sub conditioning parameters:

```bash
python main.py \
  sub_conditioning=basic \
  sub_conditioning.use_weighted_sum=true \
  [other options...]
```

## Technical Details

### How It Works

1. **Current Embedding Generation**: At each diffusion step, the current token state `x` is converted to text embeddings via `indices_to_text_embeddings(x)`

2. **Residual Calculation**: The residual `(condition - curr_embed)` is computed in the original embedding space before applying any transformations

3. **Dual Conditioning**: The DIT blocks receive both:
   - Processed current embedding (`curr_embed`)  
   - Processed residual embedding (`residual`)

4. **Modulation**: These embeddings modulate attention and MLP layers through separate linear projections with optional learnable weights

### Bug Fix

The previous implementation had a critical bug where the residual was calculated after processing `curr_embed` but before processing `condition`. This has been fixed to ensure both embeddings are in the same space when computing the residual.

**Before (buggy):**
```python
curr_embed = F.silu(self.cond_embed(curr_embed))
residual = condition - curr_embed  # Wrong: different spaces!
```

**After (fixed):**
```python
residual = condition - curr_embed  # Correct: same space
curr_embed = F.silu(self.cond_embed(curr_embed))
residual = F.silu(self.cond_embed(residual))
```

## Configuration Files

- `configs/sub_conditioning/disabled.yaml` - Baseline configuration
- `configs/sub_conditioning/basic.yaml` - Basic residual modulation  
- `configs/sub_conditioning/enabled.yaml` - Full sub conditioning

## Scripts

- `my_scripts/train_sub_conditioning.sh` - Training script
- `my_scripts/sample_sub_conditioning.sh` - Sampling/evaluation script
- `my_scripts/test_sub_conditioning.sh` - Conditioning dependence test

## Backward Compatibility

The implementation maintains backward compatibility with the old `dit.use_residual_modulation` and `dit.use_weighted_sum` configuration parameters, but the new `sub_conditioning` section is preferred.
