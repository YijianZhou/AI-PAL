"""Configurable 1D Res-U-Net for positive-window PhaseNet-style picking.

The default parameters reproduce the modernized PHN structure described for
picker_RUN: input [B, 3, 2500], dense output logits [B, 3, 2500].
"""
import torch
import torch.nn as nn
try:
    from . import config
except ImportError:
    import config

cfg = config.Config()
NUM_PHASE_CLASSES = 3


def cfg_value(name, default):
    return getattr(cfg, name, default)


def make_activation():
    return nn.GELU()


def make_norm(num_channels):
    return nn.BatchNorm1d(num_channels)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            make_norm(out_channels),
            make_activation(),
            nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            make_norm(out_channels),
        )
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(out_channels),
            )
        self.out_act = make_activation()

    def forward(self, x):
        return self.out_act(self.main(x) + self.skip(x))


class DownConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            make_norm(out_channels),
            make_activation(),
        )

    def forward(self, x):
        return self.layers(x)


class UpConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=False),
            make_norm(out_channels),
            make_activation(),
        )

    def forward(self, x):
        return self.layers(x)


class ResUNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_channel = int(cfg.num_chn)
        self.n_class = NUM_PHASE_CLASSES
        self.drop_rate = float(cfg_value('run_drop_rate', cfg_value('drop_rate', 0.0)))

        self.stage_channels = list(cfg_value('run_stage_channels', [16, 32, 64, 128, 256]))
        self.stage_kernels = list(cfg_value('run_stage_kernels', [9, 9, 7, 5, 3]))
        self.stage_blocks = list(cfg_value('run_stage_blocks', [1, 1, 1, 1, 1]))
        self.down_channels = list(cfg_value('run_down_channels', [32, 64, 128, 256, 256]))
        self.up_channels = list(cfg_value('run_up_channels', [256, 128, 64, 32, 16]))
        self.decoder_kernels = list(cfg_value('run_decoder_kernels', [3, 5, 7, 9, 9]))
        self.decoder_blocks = list(cfg_value('run_decoder_blocks', [1, 1, 1, 1, 1]))
        self.up_output_padding = list(cfg_value('run_up_output_padding', [0, 0, 0, 1, 1]))
        self.bottleneck_channels = int(cfg_value('run_bottleneck_channels', 256))
        self.bottleneck_kernel = int(cfg_value('run_bottleneck_kernel', 3))
        self.bottleneck_blocks = int(cfg_value('run_bottleneck_blocks', 1))

        self.down_kernel = int(cfg_value('run_down_kernel', 5))
        self.down_stride = int(cfg_value('run_down_stride', 2))
        self.down_padding = int(cfg_value('run_down_padding', 2))
        self.up_kernel = int(cfg_value('run_up_kernel', 5))
        self.up_stride = int(cfg_value('run_up_stride', 2))
        self.up_padding = int(cfg_value('run_up_padding', 2))

        self._validate_config()

        stem_channels = int(cfg_value('run_stem_channels', self.stage_channels[0]))
        stem_kernel = int(cfg_value('run_stem_kernel', 15))
        stem_padding = int(cfg_value('run_stem_padding', stem_kernel // 2))
        self.stem = nn.Sequential(
            nn.Conv1d(self.n_channel, stem_channels, kernel_size=stem_kernel, stride=1, padding=stem_padding, bias=False),
            make_norm(stem_channels),
            make_activation(),
        )

        self.encoder_blocks = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        in_channels = stem_channels
        for idx, channels in enumerate(self.stage_channels):
            blocks = []
            for block_idx in range(self.stage_blocks[idx]):
                blocks.append(ResidualBlock1D(
                    in_channels if block_idx == 0 else channels,
                    channels,
                    self.stage_kernels[idx],
                    drop_rate=self.drop_rate,
                ))
            self.encoder_blocks.append(nn.Sequential(*blocks))
            self.down_layers.append(DownConv1D(
                channels,
                self.down_channels[idx],
                kernel_size=self.down_kernel,
                stride=self.down_stride,
                padding=self.down_padding,
            ))
            in_channels = self.down_channels[idx]

        bottleneck = []
        for block_idx in range(self.bottleneck_blocks):
            bottleneck.append(ResidualBlock1D(
                in_channels if block_idx == 0 else self.bottleneck_channels,
                self.bottleneck_channels,
                self.bottleneck_kernel,
                drop_rate=self.drop_rate,
            ))
        self.bottleneck = nn.Sequential(*bottleneck)

        self.up_layers = nn.ModuleList()
        self.decoder_blocks_mod = nn.ModuleList()
        in_channels = self.bottleneck_channels
        for idx, out_channels in enumerate(self.up_channels):
            self.up_layers.append(UpConv1D(
                in_channels,
                out_channels,
                kernel_size=self.up_kernel,
                stride=self.up_stride,
                padding=self.up_padding,
                output_padding=self.up_output_padding[idx],
            ))
            skip_channels = self.stage_channels[::-1][idx]
            dec_kernel = self.decoder_kernels[idx]
            dec_blocks = []
            dec_in = out_channels + skip_channels
            for block_idx in range(self.decoder_blocks[idx]):
                dec_blocks.append(ResidualBlock1D(
                    dec_in if block_idx == 0 else out_channels,
                    out_channels,
                    dec_kernel,
                    drop_rate=self.drop_rate,
                ))
            self.decoder_blocks_mod.append(nn.Sequential(*dec_blocks))
            in_channels = out_channels

        self.output_conv = nn.Conv1d(self.up_channels[-1], self.n_class, kernel_size=int(cfg_value('run_output_kernel', 1)), bias=True)
        self.apply(self.init_weights)

    def _validate_config(self):
        n = len(self.stage_channels)
        fields = {
            'run_stage_kernels': self.stage_kernels,
            'run_stage_blocks': self.stage_blocks,
            'run_down_channels': self.down_channels,
            'run_up_channels': self.up_channels,
            'run_decoder_kernels': self.decoder_kernels,
            'run_decoder_blocks': self.decoder_blocks,
            'run_up_output_padding': self.up_output_padding,
        }
        for name, values in fields.items():
            if len(values) != n:
                raise ValueError('{} length {} must match run_stage_channels length {}'.format(name, len(values), n))
        if self.down_channels[-1] != self.bottleneck_channels:
            raise ValueError('Default structure expects final down channel {} to equal bottleneck channel {}'.format(
                self.down_channels[-1], self.bottleneck_channels))
        if self.up_channels[-1] != self.stage_channels[0]:
            raise ValueError('Final decoder channel should recover the stem/stage-0 width for this structure')

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def match_length(x, target_len):
        if x.size(-1) == target_len:
            return x
        if x.size(-1) > target_len:
            d = x.size(-1) - target_len
            return x[:, :, d // 2:d // 2 + target_len]
        pad_total = target_len - x.size(-1)
        left = pad_total // 2
        right = pad_total - left
        return torch.nn.functional.pad(x, (left, right))

    @staticmethod
    def align_to_skip(x, skip):
        if x.size(-1) == skip.size(-1):
            return x, skip
        if x.size(-1) > skip.size(-1):
            d = x.size(-1) - skip.size(-1)
            x = x[:, :, d // 2:d // 2 + skip.size(-1)]
        else:
            d = skip.size(-1) - x.size(-1)
            skip = skip[:, :, d // 2:d // 2 + x.size(-1)]
        return x, skip

    def forward(self, x):
        input_len = x.size(-1)
        skips = []
        x = self.stem(x)
        for enc, down in zip(self.encoder_blocks, self.down_layers):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up, dec in zip(self.up_layers, self.decoder_blocks_mod):
            x = up(x)
            skip = skips.pop()
            x, skip = self.align_to_skip(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        logits = self.output_conv(x)
        return self.match_length(logits, input_len)


class UNet(ResUNet1D):
    """Backward-compatible model name used by train/picker code."""
    pass


def model_summary():
    return {
        'stage_channels': list(cfg_value('run_stage_channels', [16, 32, 64, 128, 256])),
        'stage_kernels': list(cfg_value('run_stage_kernels', [9, 9, 7, 5, 3])),
        'stage_blocks': list(cfg_value('run_stage_blocks', [1, 1, 1, 1, 1])),
        'down_channels': list(cfg_value('run_down_channels', [32, 64, 128, 256, 256])),
        'up_channels': list(cfg_value('run_up_channels', [256, 128, 64, 32, 16])),
        'decoder_kernels': list(cfg_value('run_decoder_kernels', [3, 5, 7, 9, 9])),
        'decoder_blocks': list(cfg_value('run_decoder_blocks', [1, 1, 1, 1, 1])),
        'up_output_padding': list(cfg_value('run_up_output_padding', [0, 0, 0, 1, 1])),
    }
