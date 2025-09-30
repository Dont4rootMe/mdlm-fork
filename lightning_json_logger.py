import os
import json
import time
from typing import Dict, Any, Optional
from collections import defaultdict
import threading
import queue
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np

try:
    import lightning as L
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.utilities import rank_zero_only
    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        import pytorch_lightning as L
        from pytorch_lightning.loggers import Logger
        from pytorch_lightning.utilities import rank_zero_only
        LIGHTNING_AVAILABLE = True
    except ImportError:
        # Fallback for environments without Lightning
        LIGHTNING_AVAILABLE = False
        def rank_zero_only(func):
            """Dummy decorator when Lightning is not available."""
            return func
        
        class Logger:
            """Dummy Logger base class when Lightning is not available."""
            pass


class LightningJSONLogger(Logger if LIGHTNING_AVAILABLE else object):
    """Lightning-compatible JSON logger that saves metrics to JSONL and creates dynamic plots."""
    
    def __init__(self, save_dir: str, experiment_name: str, version: Optional[str] = None, 
                 update_freq: int = 5, max_metrics_in_memory: int = 10000):
        if LIGHTNING_AVAILABLE:
            super().__init__()
        self._save_dir = save_dir
        self._experiment_name = experiment_name
        self._version = version or "0"
        self._update_freq = update_freq  # Update plots every N steps
        self._max_metrics_in_memory = max_metrics_in_memory
        
        # Create experiment directory - if save_dir is already hydra.run.dir, use it directly with version
        # Otherwise, create nested structure
        if os.path.basename(save_dir) == experiment_name or 'outputs' in save_dir:
            # Hydra run dir case - just add version subdirectory
            self._experiment_dir = os.path.join(save_dir, f"version_{self._version}")
        else:
            # Traditional case - create full nested structure
            self._experiment_dir = os.path.join(save_dir, experiment_name, f"version_{self._version}")
        os.makedirs(self._experiment_dir, exist_ok=True)
        
        # Log files for train and validation
        self._train_log_file = os.path.join(self._experiment_dir, "train_metrics.jsonl")
        self._val_log_file = os.path.join(self._experiment_dir, "val_metrics.jsonl")
        self._config_file = os.path.join(self._experiment_dir, "config.json")
        self._plot_dir = os.path.join(self._experiment_dir, "plots")
        os.makedirs(self._plot_dir, exist_ok=True)
        
        # Initialize metrics storage for plotting
        self._metrics = defaultdict(list)
        self._step_count = 0
        self._last_plot_update = 0
        
        # Thread-safe plotting queue
        self._plot_queue = queue.Queue()
        self._plot_thread = None
        self._stop_plotting = threading.Event()
        
        # Initialize plots
        self._setup_plots()
        self._start_plot_thread()
        
    def _setup_plots(self):
        """Initialize matplotlib plots."""
        self._fig, self._axes = plt.subplots(2, 3, figsize=(15, 10))
        self._fig.suptitle(f'Training Metrics - {self._experiment_name}')
        plt.tight_layout()
        
        # Configure subplots - these will be dynamically adjusted based on actual metrics
        self._metric_axes_map = {}
        
    def _start_plot_thread(self):
        """Start background thread for plot updates."""
        self._plot_thread = threading.Thread(target=self._plot_worker, daemon=True)
        self._plot_thread.start()
        
    def _plot_worker(self):
        """Background worker for updating plots."""
        while not self._stop_plotting.is_set():
            try:
                # Wait for plot update request with timeout
                self._plot_queue.get(timeout=1.0)
                self._update_plots_impl()
                # Clear any additional queued requests
                while not self._plot_queue.empty():
                    try:
                        self._plot_queue.get_nowait()
                    except queue.Empty:
                        break
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Warning: Plot worker error: {e}")
                
    def _request_plot_update(self):
        """Request a plot update (non-blocking)."""
        try:
            self._plot_queue.put_nowait(True)
        except queue.Full:
            pass  # Skip if queue is full
        
    @property
    def name(self) -> str:
        return self._experiment_name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def save_dir(self) -> str:
        return self._save_dir
    
    @property
    def log_dir(self) -> str:
        return self._experiment_dir
    
    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log hyperparameters."""
        hparams_file = os.path.join(self._experiment_dir, "hparams.json")
        with open(hparams_file, 'w') as f:
            json.dump(params, f, indent=2, default=str)
    
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to JSONL file and update plots."""
        if step is None:
            step = self._step_count
            self._step_count += 1
        
        timestamp = time.time()
        
        # Convert tensor values to float
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = float(value.item())
            elif isinstance(value, (int, float)):
                processed_metrics[key] = float(value)
            else:
                processed_metrics[key] = str(value)
        
        # Separate train and validation metrics
        train_metrics = {}
        val_metrics = {}
        other_metrics = {}
        
        for key, value in processed_metrics.items():
            if key.startswith('train/') or key.startswith('train_'):
                clean_key = key[6:] if key.startswith('train/') else key[6:]
                train_metrics[clean_key] = value
            elif key.startswith('val/') or key.startswith('val_'):
                clean_key = key[4:] if key.startswith('val/') else key[4:]
                val_metrics[clean_key] = value
            else:
                other_metrics[key] = value
        
        # Log train metrics
        if train_metrics:
            log_entry = {
                'step': step,
                'phase': 'train',
                'timestamp': timestamp,
                'metrics': train_metrics
            }
            with open(self._train_log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        
        # Log validation metrics
        if val_metrics:
            log_entry = {
                'step': step,
                'phase': 'val',
                'timestamp': timestamp,
                'metrics': val_metrics
            }
            with open(self._val_log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        
        # Log other metrics to train file by default
        if other_metrics:
            log_entry = {
                'step': step,
                'phase': 'other',
                'timestamp': timestamp,
                'metrics': other_metrics
            }
            with open(self._train_log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
        
        # Update internal storage for plotting (with memory management)
        for key, value in processed_metrics.items():
            self._metrics[key].append((step, value))
            # Keep only recent metrics in memory
            if len(self._metrics[key]) > self._max_metrics_in_memory:
                self._metrics[key] = self._metrics[key][-self._max_metrics_in_memory:]
        
        # Update plots more frequently
        if step % self._update_freq == 0 or step - self._last_plot_update >= self._update_freq:
            self._last_plot_update = step
            self._request_plot_update()
    
    def _update_plots_impl(self):
        """Internal implementation of plot updates (called by plot worker thread)."""
        try:
            # Get all unique metric names (without train/val prefix)
            metric_names = set()
            for key in self._metrics.keys():
                if key.startswith('train/'):
                    clean_key = key[6:]  # Remove 'train/'
                    metric_names.add(clean_key)
                elif key.startswith('train_'):
                    clean_key = key[6:]  # Remove 'train_'
                    metric_names.add(clean_key)
                elif key.startswith('val/'):
                    clean_key = key[4:]  # Remove 'val/'
                    metric_names.add(clean_key)
                elif key.startswith('val_'):
                    clean_key = key[4:]  # Remove 'val_'
                    metric_names.add(clean_key)
                else:
                    metric_names.add(key)
            
            metric_names = sorted(list(metric_names))
            
            if not metric_names:
                return
            
            # Clear all axes
            for ax in self._axes.flat:
                ax.clear()
            
            # Plot up to 6 metrics (2x3 grid)
            if len(metric_names) > 6:
                print(f"Warning: Found {len(metric_names)} metrics, but only showing first 6 in plots: {metric_names[:6]}")
            
            for i, metric_name in enumerate(metric_names[:6]):
                row, col = divmod(i, 3)
                ax = self._axes[row, col]
                
                # Try different key formats
                train_keys = [f"train/{metric_name}", f"train_{metric_name}"]
                val_keys = [f"val/{metric_name}", f"val_{metric_name}"]
                
                has_data = False
                
                # Plot training metrics
                for train_key in train_keys:
                    if train_key in self._metrics and len(self._metrics[train_key]) > 0:
                        steps, values = zip(*self._metrics[train_key])
                        ax.plot(steps, values, label='Train', color='blue', alpha=0.7, linewidth=1.5)
                        has_data = True
                        break
                
                # Plot validation metrics
                for val_key in val_keys:
                    if val_key in self._metrics and len(self._metrics[val_key]) > 0:
                        steps, values = zip(*self._metrics[val_key])
                        ax.plot(steps, values, label='Validation', color='red', alpha=0.7, linewidth=1.5)
                        has_data = True
                        break
                
                # Plot other metrics (no prefix)
                if metric_name in self._metrics and len(self._metrics[metric_name]) > 0:
                    steps, values = zip(*self._metrics[metric_name])
                    ax.plot(steps, values, label='Other', color='green', alpha=0.7, linewidth=1.5)
                    has_data = True
                
                if has_data:
                    ax.set_title(metric_name.replace('_', ' ').title(), fontsize=10)
                    ax.set_xlabel('Step', fontsize=9)
                    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
                else:
                    ax.set_visible(False)
            
            # Hide unused subplots
            for i in range(len(metric_names), 6):
                row, col = divmod(i, 3)
                self._axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # Save only the latest plot (no timestamped intermediate plots)
            plot_path = os.path.join(self._plot_dir, "training_metrics_latest.png")
            self._fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                
        except Exception as e:
            print(f"Warning: Failed to update plots: {e}")
            
    def _update_plots(self):
        """Public interface for plot updates (for backward compatibility)."""
        self._request_plot_update()
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration (similar to wandb.config)."""
        try:
            with open(self._config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            print(f"Configuration saved to: {self._config_file}")
        except Exception as e:
            print(f"Warning: Failed to save config: {e}")
    
    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """Finalize logging and save final plots."""
        try:
            # Stop plot thread
            self._stop_plotting.set()
            if self._plot_thread and self._plot_thread.is_alive():
                self._plot_thread.join(timeout=5.0)
            
            # Final plot update
            self._update_plots_impl()
            final_plot_path = os.path.join(self._plot_dir, "final_training_metrics.png")
            self._fig.savefig(final_plot_path, dpi=300, bbox_inches='tight')
            
            # Save summary statistics
            self._save_summary()
            
            plt.close(self._fig)
            print(f"Final plots saved to: {final_plot_path}")
            print(f"Training logs saved to: {self._experiment_dir}")
        except Exception as e:
            print(f"Warning: Failed to save final plots: {e}")
    
    def _save_summary(self):
        """Save summary statistics."""
        try:
            summary = {}
            for key, values in self._metrics.items():
                if values:
                    steps, vals = zip(*values)
                    summary[key] = {
                        'final_value': vals[-1],
                        'min_value': min(vals),
                        'max_value': max(vals),
                        'mean_value': sum(vals) / len(vals),
                        'total_steps': len(vals)
                    }
            
            summary_file = os.path.join(self._experiment_dir, "summary.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save summary: {e}")
    
    def save(self) -> None:
        """Save the logger state."""
        pass  # Nothing to save for JSON logger
    
    def after_save_checkpoint(self, checkpoint_callback) -> None:
        """Called after checkpoint is saved."""
        pass
        
    def __del__(self):
        """Cleanup when logger is destroyed."""
        try:
            if hasattr(self, '_stop_plotting'):
                self._stop_plotting.set()
            if hasattr(self, '_plot_thread') and self._plot_thread and self._plot_thread.is_alive():
                self._plot_thread.join(timeout=1.0)
            if hasattr(self, '_fig'):
                plt.close(self._fig)
        except:
            pass
