# Python standard library
import os
import json
from typing import Dict, Tuple, Union, Any

# Third party libraries
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler
from torch.nn.functional import cross_entropy
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from optimi import StableAdamW, Lion

from architecture.encoder import Encoder
from architecture.decoder import Decoder

from utils import DatasetDDP, BatchEncoding

from diffusion_utils.corruption import apply_corruption, prepare_corruption


def cross_entropy_loss(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    loss = cross_entropy(input=input.reshape(-1, input.shape[-1]), target=target.reshape(-1), reduction="none")
    return (loss * mask.reshape(-1)).sum() / max(mask.sum(), 1)


def accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    pred = torch.argmax(logits, dim=-1)
    acc_tensor = (pred == target) * 1.
    acc = (acc_tensor * mask).sum() / max(mask.sum(), 1)
    return acc


def mse_loss_function(input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    loss = torch.mean((input - target) ** 2, dim=-1)
    return (loss * mask).sum() / max(mask.sum(), 1)
    

def to_str(list_of_tokens):
    return ",".join(str(t) for t in list_of_tokens)


def total_variation_loss(img):
     bs_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,1:,:]-img[:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,1:]-img[:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*h_img*w_img)


def kl_divergence(latent):
    """
    latent: (batch_size, latent_dim)
    It's supposed to be a normal distribution with constant variance, so only mean is used.
    """
    return 0.5 * torch.mean(latent ** 2)


class EncoderTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.step = 0

        # Initialize Accelerator for distributed training
        from accelerate import Accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=cfg.model.gradient_accumulation_steps)

        # Initialize tokenizer and set vocab configs
        # self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.autoencoder.model.text_encoder)
        from dataloader import get_tokenizer
        self.tokenizer = get_tokenizer(self.cfg)
        self.vocab_size = self.tokenizer.vocab_size

        self.device = self.accelerator.device
        
        # Configure encoder
        self._setup_encoder_cfg()
        self.encoder = Encoder(self.cfg.encoder).cuda()
        
        # Configure decoder cfg
        self._setup_decoder_cfg()
        self.decoder = Decoder(self.cfg.decoder).cuda()

        if self.cfg.training == "autoencoder":
            # Initialize training components
            self._setup_optimizer()
            self._setup_scheduler()
            self._setup_grad_scaler()

            is_loaded = self.load_checkpoint()

            # Log parameter counts
            self._log_parameter_counts()
        else:
            self.restore_checkpoint()
        
        # Setup DDP
        # self._setup_ddp()

        if self.cfg.training == "autoencoder":
            # Initialize TensorBoard logger only on main process
            if self.accelerator.is_main_process:
                self._setup_tensorboard()
            
            if is_loaded and dist.is_initialized() and self.cfg.ddp.enabled:
                self.validate()
    
    
    def _setup_encoder_cfg(self):
        """Setup encoder cfguration."""

        self.cfg.autoencoder.model.text_encoder = self.cfg.autoencoder.model.text_encoder
        self.cfg.encoder.model.text_encoder = self.cfg.autoencoder.model.text_encoder
        self.cfg.encoder.model.text_encoder_freeze_params = self.cfg.autoencoder.model.text_encoder_freeze_params
        self.cfg.encoder.tokens.vocab_size = self.vocab_size

    def _setup_decoder_cfg(self):
        """Setup decoder cfguration."""

        self.cfg.decoder.model.text_encoder = self.cfg.autoencoder.model.text_encoder
        self.cfg.decoder.model.text_encoder_freeze_params = self.cfg.autoencoder.model.text_encoder_freeze_params
        self.cfg.decoder.tokens.vocab_size = self.vocab_size
        self.cfg.decoder.tokens.mask_token_id = self.tokenizer.mask_token_id

    # def _setup_ddp(self):
    #     """Setup Distributed Data Parallel."""
    #     self.encoder = self.encoder
    #     self.decoder = self.decoder
        
    #     # if self.cfg.ddp.enabled:
    #     #     self.encoder = torch.nn.parallel.DistributedDataParallel(
    #     #         self.encoder,
    #     #         device_ids=[self.cfg.ddp.local_rank],
    #     #         broadcast_buffers=False,
    #     #         find_unused_parameters=True,
    #     #     )
            
    #     #     self.decoder = torch.nn.parallel.DistributedDataParallel(
    #     #         self.decoder,
    #     #         device_ids=[self.cfg.ddp.local_rank],
    #     #         broadcast_buffers=False,
    #     #         find_unused_parameters=True,
    #     #     )
    #     # else:
    #     #     self.encoder = self.encoder
    #     #     self.decoder = self.decoder

    def _setup_optimizer(self) -> None:
        self.grad_clip_norm = self.cfg.autoencoder.optimizer.grad_clip_norm
        
        parameters_encoder = [par[1] for par in self.encoder.named_parameters() if par[1].requires_grad]
        parameters_decoder = [par[1] for par in self.decoder.named_parameters() if par[1].requires_grad]
        
        parameters = parameters_encoder + parameters_decoder
        
        if self.cfg.autoencoder.optimizer.name == "adamw":
            optimizer = AdamW(
                parameters,
                lr=self.cfg.autoencoder.optimizer.learning_rate,
                weight_decay=self.cfg.autoencoder.optimizer.weight_decay,
                betas=(self.cfg.autoencoder.optimizer.betas[0], self.cfg.autoencoder.optimizer.betas[1]),
                eps=self.cfg.autoencoder.optimizer.eps,
            )
        elif self.cfg.autoencoder.optimizer.name == "stableadam":
            optimizer = StableAdamW(
                parameters,
                lr=self.cfg.autoencoder.optimizer.learning_rate,
                weight_decay=self.cfg.autoencoder.optimizer.weight_decay,
                betas=(self.cfg.autoencoder.optimizer.betas[0], self.cfg.autoencoder.optimizer.betas[1]),
                eps=self.cfg.autoencoder.optimizer.eps,
            )
        elif self.cfg.autoencoder.optimizer.name == "lion":
            optimizer = Lion(
                parameters,
                lr=self.cfg.autoencoder.optimizer.learning_rate,
                weight_decay=self.cfg.autoencoder.optimizer.weight_decay,
                betas=(self.cfg.autoencoder.optimizer.betas[0], self.cfg.autoencoder.optimizer.betas[1]),
            )
        
        self.optimizer = optimizer

    def _setup_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.cfg.training_setup.training_iters,
            lr_min=self.cfg.autoencoder.optimizer.min_lr,
            warmup_lr_init=self.cfg.autoencoder.optimizer.warmup_lr,
            warmup_t=self.cfg.autoencoder.optimizer.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )
        
    def _setup_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()
    
    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        # Get TensorBoard log directory from config
        tensorboard_log_dir = self.cfg.tensorboard.log_dir
        
        # Create run-specific directory using run_name from config
        run_name = self.cfg.run_name
        tensorboard_run_dir = os.path.join(tensorboard_log_dir, run_name)
        
        # Ensure directory exists
        os.makedirs(tensorboard_run_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_run_dir)
        
        print(f"📊 TensorBoard logging initialized. Logs will be saved to: {tensorboard_run_dir}")
        print(f"   To view logs, run: tensorboard --logdir {tensorboard_log_dir}")

    def _setup_train_data_generator(self) -> None:
        
        from dataloader import get_dataloaders
        
        self.train_loader, self.valid_loader = get_dataloaders(self.cfg, self.tokenizer, skip_train=False, skip_valid=False, valid_seed=None)
        
        
        # if hasattr(self, 'train_dataset'):
        #     del self.train_dataset

        # if not hasattr(self, 'train_datasets_iter'):
            # self.train_datasets_iter = DatasetDDP(
        #         config=self.cfg,
        #         split="train",
        #     ).get_dataset_iter()

        # self.train_dataset = next(self.train_datasets_iter)
        # print("Dataset length:", len(self.train_dataset))

        # self.train_loader = DataLoader(
        #     self.train_dataset,
        #     num_workers=self.cfg.autoencoder.model.num_workers,
        #     batch_size=self.cfg.autoencoder.training.batch_size_per_gpu,
        #     shuffle=True,
        #     collate_fn=self.collate_fn,
        #     drop_last=True,
        # )

    # def _setup_valid_data_generator(self) -> None:
    #     if not hasattr(self, 'valid_dataset'):
    #         self.valid_dataset = next(DatasetDDP(
    #             config=self.cfg,
    #             split="test",
    #         ).get_dataset_iter())
        
    #     self.valid_loader = DataLoader(
    #         self.valid_dataset,
    #         num_workers=self.cfg.autoencoder.model.num_workers,
    #         batch_size=self.cfg.autoencoder.training.batch_size_per_gpu,
    #         collate_fn=self.collate_fn,
    #         shuffle=False,
    #         drop_last=True,
    #     )

    def _log_parameter_counts(self) -> None:
        self.cfg.autoencoder.params.text_encoder = sum(p.numel() for p in self.encoder.text_encoder.parameters() if p.requires_grad)
        self.cfg.autoencoder.params.encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) - self.cfg.autoencoder.params.text_encoder
        self.cfg.autoencoder.params.decoder = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        self.cfg.autoencoder.params.total = self.cfg.autoencoder.params.text_encoder + self.cfg.autoencoder.params.encoder + self.cfg.autoencoder.params.decoder

    def load_checkpoint(self) -> None:
        if not self.cfg.autoencoder.model.load_checkpoint:
            return False
        
        if isinstance(self.cfg.autoencoder.model.load_checkpoint, str):
            path = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.autoencoder.model.load_checkpoint)
        else:
            path = self.find_last_checkpoint()
            if path is None:
                return False
        
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.grad_scaler.load_state_dict(state_dict["scaler"])
        self.step = state_dict["step"]
        self.latent_mean = state_dict["latent_mean"].to(self.device)
        self.latent_std = state_dict["latent_std"].to(self.device)
        self.encodings_mean = state_dict["encodings_mean"].to(self.device)
        self.encodings_std = state_dict["encodings_std"].to(self.device)
        print(f"Checkpotint {self.cfg.autoencoder.model.load_checkpoint} loaded")
        return True
    
    def find_last_checkpoint(self) -> None:
        prefix_folder = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.autoencoder.model.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            return None
        
        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return None
            
        name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"
        return checkpoint_name
    
    def restore_checkpoint(self) -> None:
        if not self.cfg.autoencoder.model.load_checkpoint:
            return
        
        path = os.path.join(self.cfg.project.checkpoint_dir, self.cfg.autoencoder.model.load_checkpoint)
        state_dict = torch.load(path)
        self.step = state_dict["step"]
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.latent_mean = state_dict["latent_mean"].to(self.device)
        self.latent_std = state_dict["latent_std"].to(self.device)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.grad_scaler.load_state_dict(state_dict["scaler"])
        self.encodings_mean = state_dict["encodings_mean"].to(self.device)
        self.encodings_std = state_dict["encodings_std"].to(self.device)
        print(f"Checkpotint {self.cfg.autoencoder.model.load_checkpoint} loaded")
        
    def save_checkpoint(self) -> None:
        # Let all ranks enter; only rank 0 will *write*, but others must participate.
        self.accelerator.wait_for_everyone()

        # Build CPU, *full* state dicts in a way compatible with sharded training.
        enc_sd = self.accelerator.get_state_dict(self.encoder)          # gathers for FSDP/ZeRO
        dec_sd = self.accelerator.get_state_dict(self.decoder)

        # Optimizer state must be gathered via accelerate too if sharded
        try:
            opt_sd = self.accelerator.optimizer_state_dict(self.optimizer)
        except AttributeError:
            # Fallback if not using sharded optimizers
            opt_sd = self.optimizer.state_dict()

        # Always move tensor storages to CPU to avoid implicit device sync during serialization
        def to_cpu(d):
            return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in d.items()}

        state_dict = {
            "step": int(self.step),
            "encoder": to_cpu(enc_sd),
            "decoder": to_cpu(dec_sd),
            "optimizer": opt_sd,                         # already CPU when gathered; if not, convert similarly
            "scheduler": self.scheduler.state_dict(),
            "scaler": (self.grad_scaler.state_dict() if hasattr(self, "grad_scaler") else {}),
        }

        if getattr(self, "latent_mean", None) is not None:
            state_dict["latent_mean"] = torch.as_tensor(self.latent_mean).cpu()
        if getattr(self, "latent_std", None) is not None:
            state_dict["latent_std"] = torch.as_tensor(self.latent_std).cpu()
        if hasattr(self, "encodings_mean"):
            state_dict["encodings_mean"] = torch.as_tensor(self.encodings_mean).cpu()
        if hasattr(self, "encodings_std"):
            state_dict["encodings_std"] = torch.as_tensor(self.encodings_std).cpu()

        # Only rank 0 performs actual file I/O
        if self.accelerator.is_main_process:
            os.makedirs(self.cfg.project.checkpoint_dir, exist_ok=True)
            prefix_folder = os.path.join(
                self.cfg.project.checkpoint_dir,
                self.cfg.autoencoder.model.checkpoints_prefix,
            )
            os.makedirs(prefix_folder, exist_ok=True)

            save_path = os.path.join(prefix_folder, f"{self.step}.pth")
            # save_path = os.path.join(prefix_folder, f"last.pth")

            # Optional: avoid the (sometimes slow) zipfile serializer on NFS/SSHFS
            torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)

        # Make sure rank 0 finishes writing before others move on
        self.accelerator.wait_for_everyone()

    # def collate_fn(self, batch):
    #     texts = [sample["text_trg"] for sample in batch]

    #     tokenized_texts = self.tokenizer(
    #         texts,
    #         add_special_tokens=self.cfg.tokenizer.add_special_tokens,
    #         padding=self.cfg.tokenizer.padding,
    #         truncation=self.cfg.tokenizer.truncation,
    #         max_length=self.cfg.dataset.max_sequence_len,
    #         return_tensors=self.cfg.tokenizer.return_tensors,
    #         return_attention_mask=self.cfg.tokenizer.return_attention_mask,
    #         return_token_type_ids=self.cfg.tokenizer.return_token_type_ids,
    #     )

    #     new_batch = {}
    #     new_batch["input_ids"] = tokenized_texts["input_ids"]
    #     new_batch["attention_mask"] = tokenized_texts["attention_mask"]

    #     # Make encodings masking and noising preparation
    #     new_batch["corrupted_attention_mask"], new_batch["mask"], new_batch["alpha"], new_batch["noise"] = prepare_corruption(
    #         encodings_shape=(new_batch["input_ids"].shape[0], new_batch["input_ids"].shape[1], self.cfg.encoder.embedding.dim),
    #         attention_mask=new_batch["attention_mask"],
    #         config=self.cfg.encoder.augmentation
    #     )

    #     return BatchEncoding(new_batch)

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, Any]):
        """
        Method for logging individual metrics to TensorBoard.
        Prefer using log_data() for batched logging which is more efficient.
        """
        # Only log on main process when using accelerator
        if self.accelerator.is_main_process and hasattr(self, 'tensorboard_writer'):
            metric_key = f'{metric_name}/{loader_name}'
            
            try:
                # Convert various types to float for TensorBoard
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:  # Single element tensor
                        value = value.item()
                    else:
                        value = value.mean().item()  # Multi-element tensor
                elif isinstance(value, (np.ndarray, np.number)):
                    value = float(value)
                elif not isinstance(value, (int, float)):
                    value = float(value)
                
                self.tensorboard_writer.add_scalar(metric_key, value, self.step)
            except Exception as e:
                print(f"Warning: Failed to log to TensorBoard: {e}")

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        
        # Use Accelerator's backward method for proper gradient scaling and distributed handling
        self.accelerator.backward(loss)

        # Calculate gradient norm before clipping using accelerator's models
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        parameters = [p for p in parameters if p.grad is not None]
        
        if parameters:
            grad_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 
                2
            )
        else:
            grad_norm = torch.tensor(0.0)

        # Use Accelerator's gradient clipping
        if self.grad_clip_norm is not None:
            self.accelerator.clip_grad_norm_(parameters, self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step_update(self.step)
        
        stat_dict = {
            "lr": self.optimizer.param_groups[0]['lr'],
            "grad_norm": grad_norm.item(),  
        }
        
        return stat_dict
    
    def log_data(self, total_loss, loss_dict, stat_dict = None, is_train: bool = True):
        if is_train:
            loader_name = "train_loader"
        else:
            loader_name = "valid_loader"
        
        # Batch all metrics together for efficient logging
        all_metrics = {}
        
        # Total loss
        all_metrics[f"Total_loss/{loader_name}"] = total_loss
        
        # Losses and accuracies
        for key in loss_dict:
            all_metrics[f"{key}/{loader_name}"] = loss_dict[key]

        # Statistics
        if stat_dict is not None:
            for k, v in stat_dict.items():
                all_metrics[f"statistics/{k}"] = v
        
        # Log all metrics to TensorBoard only on main process
        if self.accelerator.is_main_process and hasattr(self, 'tensorboard_writer'):
            try:
                for metric_name, metric_value in all_metrics.items():
                    # Convert various types to float for TensorBoard
                    if isinstance(metric_value, torch.Tensor):
                        if metric_value.numel() == 1:  # Single element tensor
                            metric_value = metric_value.item()
                        else:
                            metric_value = metric_value.mean().item()  # Multi-element tensor
                    elif isinstance(metric_value, (np.ndarray, np.number)):
                        metric_value = float(metric_value)
                    elif not isinstance(metric_value, (int, float)):
                        metric_value = float(metric_value)
                    
                    self.tensorboard_writer.add_scalar(metric_name, metric_value, self.step)
            except Exception as e:
                print(f"Warning: Failed to log to TensorBoard: {e}")
    
    def train(self) -> None:
        try:
            self.restore_checkpoint()
        except:
            pass
        
        from dataloader import get_dataloaders
        self.train_loader, self.valid_loader = get_dataloaders(self.cfg, self.tokenizer, skip_train=False, skip_valid=False, valid_seed=self.cfg.project.seed + self.accelerator.process_index)
        
        self.train_range = trange(self.step + 1, self.cfg.training_setup.training_iters + 1)
        self.train_loader_iter = iter([])

        if not hasattr(self, "encodings_mean"):
            self.encodings_mean, self.encodings_std = self.get_encodings_statistics()

        self.get_latent_statistics()
        
        self.encoder.train()
        self.encoder.text_encoder.eval()
        self.decoder.train()
        
        
        (
            self.encoder,
            self.decoder,
            self.optimizer,
            self.scheduler,
            self.grad_scaler,
            self.train_loader,
            self.valid_loader
        ) = self.accelerator.prepare(
            self.encoder,
            self.decoder,
            self.optimizer,
            self.scheduler,
            self.grad_scaler,
            self.train_loader,
            self.valid_loader
        )
        
        self.train_loader_iter = iter(self.train_loader)

        for step in self.train_range:
            
            with self.accelerator.accumulate(self.encoder), self.accelerator.accumulate(self.decoder):
            
                self.step = step
                
                batch = next(self.train_loader_iter, None)
                if batch is None:
                    self.train_loader_iter = iter(self.train_loader)
                    batch = next(self.train_loader_iter, None)
                    assert batch is not None
                    
                batch = self.batch_to_device(batch)

                total_loss, loss_dict = self.calc_loss(batch)
                stat_dict = self.optimizer_step(total_loss)

                if self.step % self.cfg.autoencoder.logging.log_freq == 0:
                    if self.accelerator.is_main_process:
                        self.log_data(total_loss, loss_dict, stat_dict, is_train=True)
                        
                        # Log model histograms less frequently (every 10x log_freq)
                        if self.step % (self.cfg.autoencoder.logging.log_freq * 10) == 0:
                            self.log_model_histograms()   
                
                self.train_range.set_description(f"total_loss: {total_loss.item():0.3f}")
                
                if self.step % self.cfg.autoencoder.logging.save_freq == 0:
                    # All ranks must participate because get_latent_statistics() uses gather_for_metrics
                    self.latent_mean, self.latent_std = self.get_latent_statistics()
                    # All ranks enter save; only rank 0 performs I/O inside save_checkpoint
                    self.save_checkpoint()


                # print(f'\n\nevaluating at step {self.step}\n\n')
                if self.step % self.cfg.autoencoder.logging.eval_freq == 0:
                    self.validate()
                    torch.cuda.empty_cache()
                # print(f'\nevaluated at step {self.step}\n\n')   
        # print(f'\n\nsaving final checkpoint at step {self.step}\n\n')
        self.latent_mean, self.latent_std = self.get_latent_statistics()
        self.save_checkpoint()
        # print(f'\n\nsaved final checkpoint at step {self.step}\n\n')

        # Finish logging on main process
        if self.accelerator.is_main_process:
            # Close TensorBoard writer
            if hasattr(self, 'tensorboard_writer'):
                self.tensorboard_writer.close()
                print("📊 TensorBoard logging finished.")   

    @torch.no_grad()
    def validate(self) -> None:
        self.encoder.eval()
        self.decoder.eval()
        

        total_loss = torch.Tensor([0.0])
        valid_dict: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        for batch in self.valid_loader:
            batch_size = batch["input_ids"].shape[0]
            batch_loss, loss_dict = self.calc_loss(batch)
            
            for k in loss_dict:
                if k in valid_dict:
                    valid_dict[k] += loss_dict[k] * batch_size
                else:
                    valid_dict[k] = torch.Tensor([loss_dict[k] * batch_size])
            valid_count += batch_size

            total_loss += batch_loss.item() * batch_size

        
        # Use Accelerator's gather_for_metrics to handle distributed accumulation
        valid_count = self.accelerator.gather_for_metrics(valid_count.cuda()).sum()
        total_loss = self.accelerator.gather_for_metrics(total_loss.cuda()).sum()
        
        for k in valid_dict:
            valid_dict[k] = self.accelerator.gather_for_metrics(valid_dict[k].cuda()).sum() / valid_count
        
        total_loss = total_loss / valid_count

        for k in valid_dict:
                valid_dict[k] = valid_dict[k] / valid_count
        
        self.log_data(total_loss, valid_dict, is_train=False)

        self.decoder.train()
        self.encoder.train()
        try:
            self.accelerator.unwrap_model(self.encoder).text_encoder.eval()
        except:
            self.encoder.text_encoder.eval()

    def get_latent(self, batch, bert_output_masking: bool = False):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # get bert hidden state
            with torch.no_grad():
                bert_hidden_state = self.accelerator.unwrap_model(self.encoder).text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state

                bert_hidden_state = self.normalize_encodings(bert_hidden_state)

            # masking bert hidden state
            if bert_output_masking:
                corrupted_bert_hidden_state = apply_corruption(
                    encodings=bert_hidden_state.detach().clone(),
                    mask=batch["attention_mask"],
                    alpha=batch["alpha"],
                    noise=batch["noise"]
                )
                attention_mask_after_corruption = batch["corrupted_attention_mask"]


                # get latents
                encoder_latents = self.encoder(
                    token_ids=input_ids,
                    mask_tokens=attention_mask_after_corruption,
                    token_embeddings=corrupted_bert_hidden_state
                )
            else:
                encoder_latents = self.encoder(
                    token_ids=input_ids,
                    mask_tokens=attention_mask,
                    token_embeddings=bert_hidden_state
                )
        return encoder_latents, bert_hidden_state
            

    def calc_loss(self, batch) -> Tuple[Dict[str, torch.Tensor]]:
        
        # if self.accelerator.is_main_process:
        #     print('\n\n\n\n')
        #     print(f"batch: {self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)}")
        #     print('\n\n\n\n')
            
        # exit()
        
        # batch = batch.to(self.device)
        batch = self.batch_to_device(batch)
        if self.cfg.suffix == "v1.0":
            latents, bert_hidden_state = self.get_latent(batch, bert_output_masking=False)
        elif self.cfg.suffix == "v2.0":
            latents, bert_hidden_state = self.get_latent(batch, bert_output_masking=False)
        else:
            latents, bert_hidden_state = self.get_latent(batch, bert_output_masking=True)

        # Corrupt latents
        if self.cfg.suffix == "final":
            p = self.cfg.encoder.augmentation.latent_masking.probability
            latents = latents * (torch.rand_like(latents) > p)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, hidden_state_of_decoder = self.decoder(
                encoder_latents=latents, 
                return_last_hidden_state=True,
            )
        
        # Compute loss
        seq_len = batch["input_ids"].shape[1]
        ce_loss = cross_entropy_loss(
            input=logits[:, :seq_len],
            target=batch["input_ids"],
            mask=batch["attention_mask"],
        )
        mse_loss = mse_loss_function(
            input=hidden_state_of_decoder[:, :seq_len],
            target=bert_hidden_state.detach().clone(),
            mask=batch["attention_mask"],
        )
        variation_loss = total_variation_loss(latents)
        if self.cfg.suffix == "v1.0":
            total_loss = ce_loss
        else:
            total_loss = ce_loss + mse_loss
        
        # Logging
        stat_dict = {}
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            stat_dict["ce_loss"] = ce_loss.detach().item()
            stat_dict["mse_loss"] = mse_loss.detach().item()

            acc = accuracy(
                logits=logits[:, :seq_len],
                target=batch["input_ids"],
                mask=batch["attention_mask"]
            )
            stat_dict["accuracy"] = acc.detach().item()

            stat_dict["variation_loss"] = variation_loss.detach().item()

        return total_loss, stat_dict

    @torch.no_grad()
    def reconstruction(self, output_file):
        self.set_valid_data_generator()
        self.encoder.eval()
        self.decoder.eval()
        
        result = []
        num_latent = self.cfg.encoder.latent.num_latents
        
        for batch in self.valid_loader:
            batch = self.batch_to_device(batch)
            # batch = batch.to(self.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                encoder_latents = self.encoder(token_ids=batch["input_ids"], mask_tokens=batch["attention_mask"])
                latents = encoder_latents[:, :num_latent]
                decoder_masked_ids = torch.ones_like(batch["input_ids"], device=self.device) * self.tokenizer.mask_token_id
                
                try:
                    decoder_masked_input = self.accelerator.unwrap_model(self.encoder).text_encoder.embeddings(decoder_masked_ids).detach().clone()
                except:
                    decoder_masked_input = self.encoder.text_encoder.embeddings(decoder_masked_ids).detach().clone()
                # decoder_masked_input = self.encoder.text_encoder.embeddings(decoder_masked_ids).detach().clone()
                logits = self.decoder(latents, masked_input=decoder_masked_input)
                pred_tokens = torch.argmax(logits, dim=-1)
            
            batch_size = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]
            
            ce_loss = cross_entropy(
                input=logits.view(-1, logits.shape[-1]),
                target=batch["input_ids"].view(-1),
                reduce=False,
            )
            ce_loss = ce_loss.reshape((batch_size, seq_len))
            
            accuracy = (pred_tokens == batch["input_ids"]) * 1.
            
            target_text = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
            pred_text = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=False)
            
            for ind in range(batch_size):
                result.append(
                    {
                        "target": target_text[ind],
                        "prediction": pred_text[ind],
                        "target_tokens": to_str(batch["input_ids"][ind].tolist()),
                        "prediction_tokens": to_str(pred_tokens[ind].tolist()),
                        "loss": ce_loss[ind].mean().item(),
                        "accuracy": accuracy[ind].mean().item(),
                    }
                )
            break
        
        loss = np.mean([r["loss"] for r in result])
        accuracy = np.mean([r["accuracy"] for r in result])
        print(f"loss: {loss:0.3f}")
        print(f"accuracy: {accuracy:0.3f}")
        
        json.dump(result, open(output_file, "w"), indent=4)

    @torch.no_grad()
    def get_latent_statistics(self,):
        self.encoder.eval()

        num_latents = self.cfg.encoder.latent.num_latents
        latent_sum = torch.zeros((num_latents, self.cfg.encoder.latent.dim), device=self.device)
        latent_sum_of_squares = torch.zeros((num_latents, self.cfg.encoder.latent.dim), device=self.device)
        latent_count = torch.Tensor([0.0]).to(self.device)
        
        for batch in self.valid_loader:
            # batch = batch.to(self.device)
            batch = self.batch_to_device(batch)
            
            latents, _ = self.get_latent(batch, bert_output_masking=False)
            latent_sum += latents.sum(dim=0)    
            latent_sum_of_squares += (latents ** 2).sum(dim=0)
            latent_count += latents.shape[0]

        # Use Accelerator's gather_for_metrics to handle distributed accumulation
        latent_count = self.accelerator.gather_for_metrics(latent_count).sum()
        latent_sum = self.accelerator.gather_for_metrics(latent_sum).sum(dim=0)
        latent_sum_of_squares = self.accelerator.gather_for_metrics(latent_sum_of_squares).sum(dim=0)
        
        latent_mean = latent_sum / latent_count
        latent_sqr = torch.clip((latent_sum_of_squares / latent_count - latent_mean ** 2), min=1e-4)
        latent_std = torch.sqrt(latent_sqr)

        return latent_mean, latent_std
    
    def normalize_latent(self, latent):
        return (latent - self.latent_mean) / self.latent_std
    
    def denormalize_latent(self, latent):
        return latent * self.latent_std + self.latent_mean
    
    
    def batch_to_device(self, batch):
        return {
            k: v.to(self.device) for k, v in batch.items()
        }
    
    @torch.no_grad()
    def get_encodings_statistics(self,):
        self.encoder.eval()
        
        encodings_sum = torch.zeros(self.cfg.encoder.embedding.dim, device=self.device)
        encodings_sum_of_squares = torch.zeros(self.cfg.encoder.embedding.dim, device=self.device)
        encodings_count = torch.Tensor([0.0]).to(self.device)
        
        for batch in tqdm(self.valid_loader):
            batch = self.batch_to_device(batch)
            # batch = batch.to(self.device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                try:
                    bert_hidden_state = self.accelerator.unwrap_model(self.encoder).text_encoder(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    ).last_hidden_state
                except:
                    bert_hidden_state = self.encoder.text_encoder(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    ).last_hidden_state

                bert_hidden_state = bert_hidden_state.reshape(-1, bert_hidden_state.shape[-1])
                mask = batch["attention_mask"].reshape(-1).bool()
                bert_hidden_state = bert_hidden_state[mask]

            encodings_sum += bert_hidden_state.sum(dim=0)    
            encodings_sum_of_squares += (bert_hidden_state ** 2).sum(dim=0)
            encodings_count += bert_hidden_state.shape[0]
            
        
        # Use Accelerator's gather_for_metrics to handle distributed accumulation
        encodings_count = self.accelerator.gather_for_metrics(encodings_count).sum()
        encodings_sum = self.accelerator.gather_for_metrics(encodings_sum).sum(dim=0)
        encodings_sum_of_squares = self.accelerator.gather_for_metrics(encodings_sum_of_squares).sum(dim=0)

        encodings_mean = encodings_sum / encodings_count
        encodings_sqr = (encodings_sum_of_squares / encodings_count - encodings_mean ** 2)
        encodings_std = torch.sqrt(torch.clip(encodings_sqr, min=1e-4))
        return encodings_mean, encodings_std
    
    def normalize_encodings(self, encodings):
        return (encodings - self.encodings_mean) / self.encodings_std
    
    def denormalize_encodings(self, encodings):
        return encodings * self.encodings_std + self.encodings_mean
    
    def log_model_histograms(self):
        """Log model parameter histograms to TensorBoard."""
        if self.accelerator.is_main_process and hasattr(self, 'tensorboard_writer'):
            try:
                # Log encoder parameters
                for name, param in self.encoder.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.tensorboard_writer.add_histogram(f'encoder/{name}', param.data, self.step)
                        self.tensorboard_writer.add_histogram(f'encoder/{name}.grad', param.grad.data, self.step)
                
                # Log decoder parameters
                for name, param in self.decoder.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.tensorboard_writer.add_histogram(f'decoder/{name}', param.data, self.step)
                        self.tensorboard_writer.add_histogram(f'decoder/{name}.grad', param.grad.data, self.step)
            except Exception as e:
                print(f"Warning: Failed to log histograms to TensorBoard: {e}")

    def cleanup_logging(self):
        """Clean up logging resources."""
        if self.accelerator.is_main_process and hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()
            print("📊 TensorBoard writer closed.")
         
        
        
        
        