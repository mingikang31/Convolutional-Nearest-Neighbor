import torch
from torch import nn, einsum
from torch.amp import autocast
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
import math 
import numpy as np 

"""Local Attention Modules"""
class SinusoidalEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        scale_base = None,
        use_xpos = False,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

    @autocast('cuda', enabled = False)
    def forward(self, x):
        seq_len, device = x.shape[-2], x.device

        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b ... r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale ** -1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k


# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim = -1)
    return normed.type(dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = padded_x.unfold(1, forward + backward + 1,1)
    return tensors.movedim(-1,dim).flatten(dim, dim + 1)

# main class
class LocalAttention(nn.Module):
    def __init__(
        self,
        window_size,
        causal = False,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
        shared_qk = False,
        rel_pos_emb_config = None,
        dim = None,
        autopad = False,
        exact_windowsize = False,
        scale = None,
        use_rotary_pos_emb = True,
        use_xpos = False,
        xpos_scale_base = None
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim,
                use_xpos = use_xpos,
                scale_base = default(xpos_scale_base, window_size // 2)
            )

    def forward(
        self,
        q, k, v,
        mask = None,
        input_mask = None,
        attn_bias = None,
        window_size = None
    ):

        mask = default(mask, input_mask)

        assert not (exists(window_size) and not self.use_xpos), 'cannot perform window size extrapolation if xpos is not turned on'

        shape, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        # auto padding

        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        scale = default(self.scale, dim_head ** -0.5)

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        # bucketing

        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        bq = bq * scale

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = (self.window_size * self.look_backward)
            max_forward_window_size = (self.window_size * self.look_forward)
            window_mask = ((bq_k - max_forward_window_size) > bq_t) | (bq_t > (bq_k + max_backward_window_size)) | pad_mask
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)

            mask = rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size)
            mask = look_around(mask, **{**look_around_kwargs, 'pad_value': False})
            mask = rearrange(mask, '... j -> ... 1 j')
            mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregation

        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, '* n d')
        return out

"""SparseAttention Modules"""
### Code from : https://github.com/openai/sparse_attention/blob/master/attention.py ###
def get_attn_mask(n, attn_mode, local_attn_ctx=None, device='cuda'):
    if attn_mode == 'all':
        # ✓ BIDIRECTIONAL - all patches attend to all patches
        b = torch.ones(n, n, device=device)
    
    elif attn_mode == 'local':
        # ✓ BIDIRECTIONAL LOCAL - attend to nearby patches in both directions
        bandwidth = local_attn_ctx
        # Create a band matrix (not just lower triangular)
        b = torch.zeros(n, n, device=device)
        for i in range(n):
            start = max(0, i - bandwidth // 2)
            end = min(n, i + bandwidth // 2 + 1)
            b[i, start:end] = 1
    
    elif attn_mode == 'strided':
        # ✓ BIDIRECTIONAL STRIDED
        stride = local_attn_ctx
        x = torch.arange(n, dtype=torch.int32, device=device).view(n, 1)
        y = x.t()
        q = x.expand(n, n)
        k = y.expand(n, n)
        # Remove c1 = q >= k (this was the causal constraint!)
        c2 = ((q - k).abs() % stride) == 0  # Distance is multiple of stride
        b = c2.float()
    
    b = b.view(1, 1, n, n)
    return b


def strided_transpose(x, n_ctx, local_attn_ctx, blocksize=None):
    """
    Transpose for strided attention pattern.
    
    Args:
        x: tensor of shape [batch, seq_len, embd]
        n_ctx: context length
        local_attn_ctx: stride length
        blocksize: not used in PyTorch version (kept for API compatibility)
    
    Returns:
        transposed tensor
    """
    bT_ctx = n_ctx // local_attn_ctx
    n, t, embd = x.shape
    x = x.view(n, bT_ctx, local_attn_ctx, embd)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(n, t, embd)
    return x


def split_heads(x, n_heads):
    """
    Split the last dimension into (n_heads, depth).
    Transpose to shape [batch, n_heads, seq_len, depth]
    """
    batch_size, seq_len, d_model = x.shape
    depth = d_model // n_heads
    x = x.view(batch_size, seq_len, n_heads, depth)
    return x.permute(0, 2, 1, 3)


def merge_heads(x):
    """
    Merge heads back to original shape.
    Input: [batch, n_heads, seq_len, depth]
    Output: [batch, seq_len, d_model]
    """
    batch_size, n_heads, seq_len, depth = x.shape
    x = x.permute(0, 2, 1, 3)
    return x.reshape(batch_size, seq_len, n_heads * depth)


def attention_impl(q, k, v, n_heads, attention_dropout, attn_mode, local_attn_ctx=None):
    """
    Standard attention implementation with different masking patterns.
    
    Args:
        q, k, v: query, key, value tensors of shape [batch, seq_len, d_model]
        n_heads: number of attention heads
        attn_mode: attention pattern ('all', 'local', 'strided')
        local_attn_ctx: context window for local/strided attention
    
    Returns:
        attention output of shape [batch, seq_len, d_model]
    """
    # Split heads: [batch, n_heads, seq_len, depth]
    q = split_heads(q, n_heads)
    k = split_heads(k, n_heads)
    v = split_heads(v, n_heads)
    
    # Get attention mask
    n_timesteps = k.shape[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx, device=q.device)
    
    # Scaled dot-product attention
    # [batch, n_heads, seq_len, seq_len]
    depth = q.shape[-1]
    scale_amount = 1.0 / np.sqrt(depth)
    
    # Compute attention scores
    w = torch.matmul(q, k.transpose(-2, -1))
    w = w * scale_amount
    
    # Apply mask (using large negative value for masked positions)
    w = w * mask + -1e9 * (1 - mask)
    
    # Softmax
    w = F.softmax(w, dim=-1)

    w = F.dropout(w, p=attention_dropout)
    
    # Apply attention to values
    a = torch.matmul(w, v)
    
    # Merge heads
    a = merge_heads(a)
    
    return a

# For gradient checkpointing (equivalent to @recomputable decorator)
def checkpoint_attention(q, k, v, n_heads, attn_mode, local_attn_ctx=None):
    """
    Attention with gradient checkpointing to save memory.
    """
    return torch.utils.checkpoint.checkpoint(
        attention_impl,
        q, k, v, n_heads, attn_mode, local_attn_ctx,
        use_reentrant=False
    )

def strided_attention_impl(q, k, v, n_heads, local_attn_ctx, blocksize=32):
    """
    Strided attention with transposition (as in blocksparse version).
    
    Note: This is the dense implementation. For true block-sparse computation,
    you would need a custom CUDA kernel or library like Triton.
    """
    n_ctx = q.shape[1]
    
    # Apply strided transpose
    q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
    k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
    v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    
    # Apply attention
    a = attention_impl(q, k, v, n_heads, 'strided', local_attn_ctx)
    
    # Reverse the transpose
    n, t, embd = a.shape
    bT_ctx = n_ctx // local_attn_ctx
    a = a.view(n, local_attn_ctx, bT_ctx, embd)
    a = a.permute(0, 2, 1, 3)
    a = a.reshape(n, t, embd)
    
    return a


