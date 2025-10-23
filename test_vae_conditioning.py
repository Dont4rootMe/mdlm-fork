"""
Test script for VAE-style conditioning in DiT

This script demonstrates how to use the sophisticated VAE-style conditioning
mechanism implemented in dit_vae_condition.py.
"""

import torch
import omegaconf
from models.dit_vae_condition import DIT


def test_vae_conditioning():
    """Test the VAE-style conditioning mechanism"""
    
    # Create a simple config
    config = omegaconf.OmegaConf.create({
        'model': {
            'hidden_size': 768,
            'n_heads': 12,
            'n_blocks': 6,
            'cond_dim': 768,
            'dropout': 0.1,
            'scale_by_sigma': True,
            # VAE conditioning parameters (optional, will use defaults if not specified)
            'cond_encoder_layers': 2,        # Number of transformer layers in condition encoder
            'cond_encoder_hidden': 768,      # Hidden size for cross-attention
            'cond_encoder_heads': 8,         # Number of attention heads
            'cond_encoder_ff_mult': 4,       # Feed-forward multiplier
        }
    })
    
    # Model parameters
    vocab_size = 50257  # GPT-2 vocab size
    cond_dim = 768      # Condition embedding dimension
    batch_size = 2
    seq_len = 128
    
    # Initialize model
    print("Initializing DIT with VAE-style conditioning...")
    model = DIT(config, vocab_size=vocab_size, cond_dim=cond_dim)
    model.eval()
    
    # Create dummy inputs
    indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    sigma = torch.rand(batch_size)
    condition = torch.randn(batch_size, cond_dim)  # Simple conditioning vector
    
    print(f"\nInput shapes:")
    print(f"  indices: {indices.shape}")
    print(f"  sigma: {sigma.shape}")
    print(f"  condition: {condition.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(indices, sigma, condition)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, {seq_len}, {vocab_size}]")
    
    assert output.shape == (batch_size, seq_len, vocab_size), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {seq_len}, {vocab_size})"
    
    # Test with None condition (unconditional)
    print("\nTesting unconditional generation (condition=None)...")
    with torch.no_grad():
        output_uncond = model(indices, sigma, None)
    
    print(f"Unconditional output shape: {output_uncond.shape}")
    assert output_uncond.shape == (batch_size, seq_len, vocab_size)
    
    # Test with multi-dimensional condition (e.g., from text encoder)
    print("\nTesting with sequential condition...")
    cond_seq_len = 32
    condition_seq = torch.randn(batch_size, cond_seq_len, cond_dim)
    print(f"  condition_seq: {condition_seq.shape}")
    
    with torch.no_grad():
        output_seq = model(indices, sigma, condition_seq)
    
    print(f"Output with sequential condition: {output_seq.shape}")
    assert output_seq.shape == (batch_size, seq_len, vocab_size)
    
    print("\n✅ All tests passed!")
    print("\nVAE-style conditioning components:")
    print(f"  - Learnable latents: {model.cond_encoder.latents.shape}")
    print(f"  - Number of encoder layers: {len(model.cond_encoder.layers)}")
    print(f"  - Cross-attention heads: {model.cond_encoder.layers[0].cross_attn.num_heads}")
    print(f"  - Hidden size: {model.cond_encoder.layers[0].cross_attn.hidden_size}")
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    cond_encoder_params = sum(p.numel() for p in model.cond_encoder.parameters())
    print(f"\nParameter counts:")
    print(f"  - Total model parameters: {total_params:,}")
    print(f"  - Condition encoder parameters: {cond_encoder_params:,}")
    print(f"  - Condition encoder ratio: {cond_encoder_params / total_params * 100:.2f}%")


def compare_with_simple_conditioning():
    """Compare VAE-style conditioning with simple linear projection"""
    
    print("\n" + "="*80)
    print("Comparing VAE-style conditioning vs simple linear projection")
    print("="*80)
    
    config = omegaconf.OmegaConf.create({
        'model': {
            'hidden_size': 768,
            'n_heads': 12,
            'n_blocks': 6,
            'cond_dim': 768,
            'dropout': 0.1,
            'scale_by_sigma': True,
            'cond_encoder_layers': 2,
            'cond_encoder_hidden': 768,
            'cond_encoder_heads': 8,
            'cond_encoder_ff_mult': 4,
        }
    })
    
    vocab_size = 50257
    cond_dim = 768
    
    # VAE-style model
    model_vae = DIT(config, vocab_size=vocab_size, cond_dim=cond_dim)
    
    # Count parameters for conditioning
    vae_cond_params = sum(p.numel() for p in model_vae.cond_encoder.parameters())
    
    print(f"\nVAE-style conditioning:")
    print(f"  - Parameters: {vae_cond_params:,}")
    print(f"  - Architecture: Learnable latents + Cross-attention + Transformer")
    print(f"  - Capacity: High (can capture complex conditioning patterns)")
    
    print(f"\nSimple linear projection (for comparison):")
    simple_params = cond_dim * config.model.hidden_size  # Linear layer
    print(f"  - Parameters: {simple_params:,}")
    print(f"  - Architecture: Single linear transformation")
    print(f"  - Capacity: Low (limited representational power)")
    
    print(f"\nParameter ratio (VAE / Simple): {vae_cond_params / simple_params:.2f}x")
    print("\nAdvantages of VAE-style conditioning:")
    print("  ✓ Richer representation through attention mechanism")
    print("  ✓ Learnable latents adapt to conditioning distribution")
    print("  ✓ Multiple layers allow hierarchical feature extraction")
    print("  ✓ Can handle variable-length conditioning sequences")
    print("  ✓ Better generalization on complex conditioning tasks")


if __name__ == "__main__":
    print("="*80)
    print("Testing DiT with VAE-Style Conditioning")
    print("="*80)
    
    test_vae_conditioning()
    compare_with_simple_conditioning()
    
    print("\n" + "="*80)
    print("All tests completed successfully! 🎉")
    print("="*80)

