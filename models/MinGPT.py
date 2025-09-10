# Training (a smaller version) of MinGPT from scratch
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os
import math
import inspect
from sklearn.model_selection import train_test_split
from torch.utils.checkpoint import checkpoint

from data.dataset import *
from data.data_utils import get_running_device, generate_random_sample

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

class MinGPTConfig:
    """
    Custom MinGPT Training Configuration.

    Original configurations:
        'gpt2':         n_layer=12, n_head=12, n_embd=768,  # 124M params
        'gpt2-medium':  n_layer=24, n_head=16, n_embd=1024, # 350M params
        'gpt2-large':   n_layer=36, n_head=20, n_embd=1280, # 774M params
        'gpt2-xl':      n_layer=48, n_head=25, n_embd=1600, # 1558M params
    """
    def __init__(self, dataset_size='medium', use_privacy=False):
        self.device = get_running_device() # allow mps
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.vocab_size = 4104  # 4098  # tokenizer vocab size + special tokens
        self.block_size = 4676 # 64 # 512
        self.max_length = 4676 # 512 # 1024 
        self.n_layer = 6 #4 #6 #12
        self.n_head = 6 # 4 #6 #12
        self.n_embd = 384 #128 #384 #768
        self.dropout = 0.0 # 0.01
        self.bias = False   # True: bias in Linears and LayerNorms, like GPT-2. 
                            # False: better + faster
        
        # Training info
        self.batch_size = 16 # 64 # 32 # 128
        self.learning_rate = 1e-3 # 3e-4 # 5e-2 # 1e-3
        self.weight_decay = 0.1
        self.num_epochs = 2504 // self.batch_size #100
        self.save_every = self.num_epochs // 5
        self.optimizer = 'adam'  # 'adam', 'sgd', 'adam_corr'

        # Privacy info
        self.use_privacy = use_privacy
        self.target_epsilon = 4.0 # 3.0
        self.target_delta = 1e-5
        self.max_grad_norm = 1.5 #1.0
        self.noise_multiplier = 0.5

        self.train_data_path = f"data/sources/json/{dataset_size}/train_gts_with_pop.json"
        self.val_data_path = f"data/sources/json/{dataset_size}/val_gts_with_pop.json"
        if self.use_privacy:
            self.checkpoint_dir = "models/saved/minGPT/DP/checkpoints"
        else:
            self.checkpoint_dir = "models/saved/minGPT/checkpoints"

    def __str__(self):
        desc = "MinGPT Configuration"
        if self.use_privacy:
            desc += f" (using DP)"
        desc += f"\nModel Architecture:\n"
        desc += f"  Vocab size: {self.vocab_size}\n"
        desc += f"  Block Size: {self.block_size}\n"
        desc += f"  Num Layers: {self.n_layer}\n"
        desc += f"  Num Heads: {self.n_head}\n"
        desc += f"  Embedding Size: {self.n_embd}\n"
        if self.dropout != 0.0:
            desc += f"  Dropout: {self.dropout}\n"
        if self.bias:
            desc += f"  Bias Enabled.\n"
        desc += f"\nModel Max Length: {self.max_length}"
        desc += f"\nBatch Size: {self.batch_size}"
        desc += f"\nLearning Rate: {self.learning_rate}"
        desc += f"\nWeight Decay: {self.weight_decay}"
        desc += f"\nNum Epochs: {self.num_epochs}"
        desc += f"\nOptimizer: {self.optimizer}"
        if self.use_privacy:
            desc += f"\nTarget Epsilon: {self.target_epsilon}"
            desc += f"\nTarget Delta: {self.target_delta}"
            desc += f"\nMax Grad Norm: {self.max_grad_norm}"
            desc += f"\nNoise Multiplier: {self.noise_multiplier}"
        return desc

############################### MinGPT ###############################
#
# Full Credits: https://github.com/karpathy/minGPT
#
# Other References:
#   1) the official GPT-2 TensorFlow implementation released by OpenAI:
#       https://github.com/openai/gpt-2/blob/master/src/model.py
#   2) huggingface/transformers PyTorch implementation:
#       https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
#
######################################################################

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class MinGPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        if config.use_privacy:
            # Avoid shared weights with Opacus and DP
            self.transformer.wte.weight = nn.Parameter(self.lm_head.weight.clone().detach())
            # freeze pos embedding parameters to handle misshape error
            # (Opacus expects batch size to be the first dim, here it is not)
            print("Freezing position embedding parameters for DP training")
            for param in self.transformer.wpe.parameters():
                param.requires_grad = False
        else:
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless.
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("Model has %.2fM" % (self.get_num_params()/1e6,), "parameters.")

    def __str__(self):
        return super().__str__()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        if t > self.config.block_size:
            print("Resizing")
            # Use only the first block_size tokens for position embeddings
            idx = idx[:, :self.config.block_size]
            if targets is not None:
                targets = targets[:, :self.config.block_size]
            t = self.config.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # if self.config.use_privacy:
        #     # handle misshape error [FIXED: freeze weights at initialization]
        #     pos = pos.unsqueeze(0).expand(b, -1) # shape (b,t)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            # x = block(x)
            x = checkpoint(block, x, use_reentrant=True) # save memory
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, saved_model_path, **kwargs):
        """
        Load the model from a local pretrained checkpoint.
        To use for inference after model has been trained.
        """
        device = kwargs.get('device', 'cpu')
        loaded_model = torch.load(saved_model_path, weights_only=False, map_location=torch.device(device))
        return loaded_model

    def save_model_checkpoint(self, checkpoint_dir, epoch, optimizer, config, training_curves, training_log):
        """
        Save model checkpoint and training configurations (basic model).
        """
        checkpoint = {'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_curves': training_curves,
                    'config': config}

        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        training_log.write(f"Saved checkpoint: {checkpoint_path}")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
def save_dpmodel_checkpoint(model, checkpoint_dir, epoch, optimizer, config, training_curves, training_log):
        """
        Save model checkpoint and training info (DP-specific).
        Outside of the model's class methods, as with opacus the model
        is wrapped in a privacy-preserving context.
        """
        checkpoint = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_curves': training_curves,
                    'config': config}
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 
                                       f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        training_log.write(f"Saved checkpoint: {checkpoint_path}")


################################ Trainer ################################
import time
class MinGPTTrainer:
    """
    Adapted from https://github.com/karpathy/minGPT
    """
    def __init__(self, config, model, tokenizer, save_losses=True):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = get_running_device()
            # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        # print("Running on device:", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        self.save_losses = save_losses
        self.loss = 0

    def configure_optimizers(self, model):

        def get_optim_groups(model, config):
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in model.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': config.weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"    Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"    Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            return optim_groups
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.config.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # if self.config.use_privacy:
        #     optim_groups = model.parameters()
        # else:
        #     optim_groups = get_optim_groups(model, self.config)
        optim_groups = get_optim_groups(model, self.config)
        
        # Create optimizer
        if self.config.optimizer == 'adam':
            # use the fused version if it is available
            optimizer = torch.optim.AdamW(optim_groups, 
                                    lr=self.config.learning_rate, 
                                    weight_decay=self.config.weight_decay,
                                    **extra_args)
        else:
            optimizer = torch.optim.SGD(optim_groups, 
                                lr=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)

        print(f"    Using{'fused ' if use_fused else ' '}{optimizer.__class__.__name__} as optimizer")

        return optimizer

    def setup_training_data(self, dataset_sequences, train_val_split=0.3):
        
        def collate_fn(batch):
            """
            Collate function for batching custom dataset.
            """
            inputs = torch.stack([item['input_ids'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            return {'input_ids': inputs, 
                    'labels': labels}
        
        from torch.utils.data import DataLoader
        
        train_sequences, val_sequences = train_test_split(
                                dataset_sequences, 
                                test_size=train_val_split, 
                                random_state=42)
        
        train_dataset = MinGPTDataset(train_sequences, 
                                    self.tokenizer, 
                                    self.config.max_length)
        eval_dataset = MinGPTDataset(val_sequences,
                                    self.tokenizer,
                                    self.config.max_length)
        
        train_dataloader = DataLoader(train_dataset,
                                    batch_size=self.config.batch_size,
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False)
        eval_dataloader = DataLoader(eval_dataset,
                                    batch_size=self.config.batch_size,
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False)
        return train_dataloader, eval_dataloader
    
    def setup_differential_privacy(self, train_dataloader):
        print("    Setting up differential privacy:") 

        # Make model and optimizer compatible with Opacus
        model = ModuleValidator.fix(self.model)
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            print(f"    Fixing model validation errors: {errors}")
            model = ModuleValidator.fix(model)

        optimizer = self.configure_optimizers(model)

        privacy_engine = PrivacyEngine(accountant="rdp") #secure_mode=True)
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
                                        module=model,
                                        optimizer=optimizer,
                                        data_loader=train_dataloader,
                                        # embedding_layers=[model.transformer.wte,
                                        #                   model.transformer.wpe]
                                        epochs=self.config.num_epochs,
                                        target_epsilon=self.config.target_epsilon,
                                        target_delta=self.config.target_delta,
                                        max_grad_norm=self.config.max_grad_norm,
                                        grad_sample_mode="hooks") # for checkpointing
        # model, optimizer, train_dataloader = privacy_engine.make_private(
        #                                 module=model,
        #                                 optimizer=optimizer,
        #                                 data_loader=train_dataloader,
        #                                 epochs=self.config.num_epochs,
        #                                 target_epsilon=self.config.target_epsilon,
        #                                 target_delta=self.config.target_delta,
        #                                 max_grad_norm=self.config.max_grad_norm,
        #                                 noise_multiplier=self.config.noise_multiplier,
        #                                 # clipping='per_layer',
        #                                 grad_sample_mode="hooks", # for checkpointing
        #                                 embedding_layers=[model.transformer.wte,
        #                                                   model.transformer.wpe])
        
        print(f"    Privacy Budget: ε={self.config.target_epsilon}, δ={self.config.target_delta}")
        return model, optimizer, train_dataloader, privacy_engine

    def run(self, train_dataloader, eval_dataloader):
        """
        Model training pipeline with optional differential privacy support
        """
        # setup the optimizer
        if self.config.use_privacy: 
            (self.model, optimizer, train_dataloader,
            privacy_engine) = self.setup_differential_privacy(train_dataloader)
        else:
            # Initialize default optimizer without DP
            optimizer = self.configure_optimizers(self.model)

        self.iter_num = 0
        self.iter_time = time.time()
        eval_iter = iter(eval_dataloader)

        train_losses = []
        eval_losses = []

        eval_str = ""
        privacy_str = ""
        used_epsilons = []

        epochs = self.config.num_epochs
        eval_steps = max(1, epochs // 10)
        training_log = tqdm(range(epochs), position=0)
        
        eval_iter = iter(eval_dataloader)
        # Main Training Loop
        for ep in training_log:
            self.model.train()
            training_log.set_description(f"Epoch {ep+1}/{epochs}")

            for batch in iter(train_dataloader):
                xb, yb = batch['input_ids'].to(self.device), batch['labels'].to(self.device)

                optimizer.zero_grad(set_to_none=True)
                
                logits, self.loss = self.model(xb, yb)

                train_losses.append(self.loss.item())

                # backprop and update the parameters
                optimizer.zero_grad(set_to_none=True)
                self.loss.backward()
                optimizer.step()
                
                torch.cuda.empty_cache() # for memory

            # [FIXED]: to check for opacus shape mismatch error
            # for name, param in self.model.named_parameters():
            #     if hasattr(param, "grad_sample"):
            #         print(f"{name}: grad_sample shape = {param.grad_sample.shape}")
            #
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
            #                               self.config.grad_norm_clip)

            if ep % eval_steps == 0 or ep == epochs-1:
                self.model.eval()
                with torch.no_grad():
                    try:
                        batch = next(eval_iter)
                    except StopIteration:
                        print("Resetting eval iterator")
                        eval_iter = iter(eval_dataloader)
                        batch = next(eval_iter)
                    xvb, yvb = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                    _, e_loss = self.model(xvb, yvb)
                    eval_losses.append(e_loss.item())
                    eval_str = f"\teval_loss: {e_loss:.4f}"

            if self.config.use_privacy:
                epsilon = privacy_engine.get_epsilon(self.config.target_delta)
                privacy_str = f"\tε_used: {epsilon:.2f}"
                used_epsilons.append(epsilon)

            training_log.set_postfix_str( f"iter_dt {self.iter_dt * 1000:.2f}ms;"+
                                        f"\tlr: {self.config.learning_rate}" +
                                        f"\ttrain_loss: {self.loss:.4f}" +
                                        f"{eval_str}" +
                                        f"{privacy_str}")
            
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            self.saved_losses = {'train_losses': train_losses, 
                                'eval_losses': eval_losses}
            if self.config.use_privacy:
                self.used_epsilons = used_epsilons
            
            # Save checkpoint
            if (ep + 1) % self.config.save_every == 0:
                if self.config.use_privacy:
                    training_curves = {'train_losses': train_losses, 'eval_losses': eval_losses, 'used_epsilons': used_epsilons}
                    save_dpmodel_checkpoint(self.model, self.config.checkpoint_dir, ep + 1, optimizer, self.config, training_curves, training_log)
                else:
                    self.model.save_model_checkpoint(self.config.checkpoint_dir, ep + 1, optimizer, self.config, self.saved_losses, training_log)

############################## Sample Generation ##############################
def generate_synthetic_samples(model, tokenizer, num_samples=1, 
                            max_sample_length=100, **kwargs):
    """
    Generate synthetic genomic sequences using trained model
    """
    start_prompt = kwargs.get('start_prompt', None)
    device = torch.device(kwargs.get('device', 'cpu'))
    model.to(device)

    if start_prompt is None:
        # generate a random prompt if not given one
        start_prompt = generate_random_sample(num_mutations=kwargs.get('prompt_length', 1))
    # print(f"Using start prompt: {start_prompt}")

    synthetic_samples = []
    return_tensors = kwargs.get('return_tensors', False)
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            tokenized_prompt = tokenizer.encode(start_prompt, allowed_special='all')
            tokenized_prompt = torch.LongTensor(tokenized_prompt).unsqueeze(0)
            tokenized_prompt = tokenized_prompt.to(device)
            sample = model.generate(tokenized_prompt, max_sample_length)
            if not return_tensors:
                sample = tokenizer.decode(sample.tolist()[0])
            synthetic_samples.append(sample)
            # print(f"Generated sequence {i+1}: {synthetic_samples[-1][:20]}...")  # Show first 20 chars

    return synthetic_samples