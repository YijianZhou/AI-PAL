"""Frame-level Transformer picker with RoPE attention.

Class order is fixed throughout the package:
    0 = Noise, 1 = P, 2 = S

Input:  [B, 3, 2500]
Output: [B, 3, 246] logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import config
except ImportError:
    import config


cfg = config.Config()
CLASS_NAMES = ('Noise', 'P', 'S')


class RMSNorm(nn.Module):
    """Bias-free RMS normalization over the final dimension."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        source_dtype = x.dtype
        x_float = x.float()
        x_norm = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_norm * self.weight.float()).to(source_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim, max_sequence_length=512, base_theta=10000.0):
        super().__init__()
        if rotary_dim % 2:
            raise ValueError('rotary_dim must be even')
        self.rotary_dim = int(rotary_dim)
        inv_freq = 1.0 / (
            float(base_theta) ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer('cos_cached', angles.cos(), persistent=False)
        self.register_buffer('sin_cached', angles.sin(), persistent=False)

    def forward(self, q, k):
        sequence_length = q.size(-2)
        if sequence_length > self.cos_cached.size(0):
            raise ValueError(
                'Sequence length {} exceeds RoPE cache {}'.format(
                    sequence_length, self.cos_cached.size(0)
                )
            )
        cos = self.cos_cached[:sequence_length].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        sin = self.sin_cached[:sequence_length].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _rotate(self, x, cos, sin):
        rotary = x[..., :self.rotary_dim]
        passthrough = x[..., self.rotary_dim:]
        even = rotary[..., 0::2]
        odd = rotary[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        rotated = rotated.flatten(-2)
        return torch.cat((rotated, passthrough), dim=-1)


class ConvFrameEmbedding(nn.Module):
    def __init__(self, in_channels, d_model, frame_samples, frame_stride_samples, norm_eps, dropout):
        super().__init__()
        self.projection = nn.Conv1d(
            in_channels,
            d_model,
            kernel_size=frame_samples,
            stride=frame_stride_samples,
            padding=0,
            bias=True,
        )
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.projection(x).transpose(1, 2)
        return self.dropout(self.norm(x))


class RoPEMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        rotary_dim,
        max_sequence_length,
        base_theta,
        attention_dropout,
        output_dropout,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError('d_model must be divisible by num_heads')
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        if rotary_dim > self.head_dim:
            raise ValueError('rotary_dim cannot exceed head_dim')
        self.attention_dropout = float(attention_dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.rope = RotaryEmbedding(rotary_dim, max_sequence_length, base_theta)
        self.output_projection = nn.Linear(d_model, d_model, bias=True)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope(q, k)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, self.d_model)
        return self.output_dropout(self.output_projection(x))


class TransformerFFN(nn.Module):
    def __init__(self, d_model, hidden_size, hidden_dropout, output_dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size, d_model),
            nn.Dropout(output_dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        d_model = model_config['d_model']
        norm_eps = model_config['norm_eps']
        self.norm_1 = RMSNorm(d_model, eps=norm_eps)
        self.attention = RoPEMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=model_config['num_heads'],
            rotary_dim=model_config['rotary_dim'],
            max_sequence_length=model_config['max_sequence_length'],
            base_theta=model_config['base_theta'],
            attention_dropout=model_config['attention_dropout'],
            output_dropout=model_config['output_dropout'],
        )
        self.norm_2 = RMSNorm(d_model, eps=norm_eps)
        self.ffn = TransformerFFN(
            d_model=d_model,
            hidden_size=model_config['ffn_hidden'],
            hidden_dropout=model_config['ffn_dropout'],
            output_dropout=model_config['output_dropout'],
        )

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        x = x + self.ffn(self.norm_2(x))
        return x


class FrameTransformerPicker(nn.Module):
    """Dense Noise/P/S classifier over overlapping waveform frames."""

    def __init__(self):
        super().__init__()
        self.input_samples = int(round(cfg.win_len * cfg.samp_rate))
        self.frame_samples = int(round(cfg.ft_frame_length * cfg.samp_rate))
        self.frame_stride_samples = int(round(cfg.ft_frame_step * cfg.samp_rate))
        self.num_frames = (self.input_samples - self.frame_samples) // self.frame_stride_samples + 1
        model_config = {
            'd_model': int(cfg.ft_d_model),
            'num_heads': int(cfg.ft_num_heads),
            'rotary_dim': int(cfg.ft_rotary_dim),
            'max_sequence_length': int(cfg.ft_max_sequence_length),
            'base_theta': float(cfg.ft_rope_base_theta),
            'attention_dropout': float(cfg.ft_attention_dropout),
            'output_dropout': float(cfg.ft_output_dropout),
            'ffn_hidden': int(cfg.ft_ffn_hidden),
            'ffn_dropout': float(cfg.ft_ffn_dropout),
            'norm_eps': float(cfg.ft_norm_eps),
        }
        if model_config['max_sequence_length'] < self.num_frames:
            raise ValueError('ft_max_sequence_length must cover all {} frames'.format(self.num_frames))
        self.frame_embedding = ConvFrameEmbedding(
            in_channels=int(cfg.num_chn),
            d_model=model_config['d_model'],
            frame_samples=self.frame_samples,
            frame_stride_samples=self.frame_stride_samples,
            norm_eps=model_config['norm_eps'],
            dropout=float(cfg.ft_embedding_dropout),
        )
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(model_config) for _ in range(int(cfg.ft_num_layers))
        ])
        self.final_norm = RMSNorm(model_config['d_model'], eps=model_config['norm_eps'])
        self.classifier = nn.Linear(model_config['d_model'], 3, bias=True)

    def forward(self, x):
        if x.ndim != 3 or x.size(1) != int(cfg.num_chn):
            raise ValueError('Expected waveform [B, {}, N], got {}'.format(cfg.num_chn, tuple(x.shape)))
        if x.size(-1) != self.input_samples:
            raise ValueError('Expected {} input samples, got {}'.format(self.input_samples, x.size(-1)))
        x = self.frame_embedding(x)
        for block in self.blocks:
            x = block(x)
        logits = self.classifier(self.final_norm(x))
        return logits.transpose(1, 2).contiguous()


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


