import torch
import torch.nn as nn
try:
    from . import config
except ImportError:
    import config
cfg = config.Config()
NUM_PHASE_CLASSES = 3


def get_padding(kernel_size, stride):
    return (kernel_size - stride + 1) // 2


def get_output_padding(input_len, exp_output_len, kernel_size, stride, padding):
    output_len = (input_len - 1) * stride - 2 * padding + kernel_size
    return exp_output_len - output_len


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, transpose=False,
                 output_padding=0, bias=False, drop_rate=0.0):
        super().__init__()
        if transpose:
            conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding,
                                      output_padding=output_padding, bias=bias)
        else:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=bias)
        layers = [conv, nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True)]
        if drop_rate > 0:
            layers.append(nn.Dropout(p=drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.depths = cfg.depths
        self.filters_root = cfg.filters_root
        self.kernel_size = cfg.kernel_size
        self.pool_size = cfg.pool_size
        self.n_channel = cfg.num_chn
        self.n_class = NUM_PHASE_CLASSES
        self.drop_rate = getattr(cfg, 'drop_rate', 0.0)
        self.encoder_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        input_len = int(cfg.samp_rate * cfg.win_len)

        padding = get_padding(self.kernel_size, 1)
        self.input_layers = ConvBNReLU(self.n_channel, self.filters_root,
                                       self.kernel_size, padding=padding,
                                       bias=True, drop_rate=self.drop_rate)

        num_filters = self.filters_root
        down_samp_len = input_len
        encode_len = [down_samp_len]

        for depth in range(self.depths):
            in_channels = num_filters
            num_filters = int(2**depth * self.filters_root)
            padding = get_padding(self.kernel_size, 1)
            self.encoder_layers.append(
                ConvBNReLU(in_channels, num_filters, self.kernel_size,
                           padding=padding, bias=False, drop_rate=self.drop_rate))
            if depth == self.depths - 1:
                continue

            padding = get_padding(self.kernel_size, self.pool_size)
            self.pool_layers.append(
                ConvBNReLU(num_filters, num_filters, self.kernel_size,
                           stride=self.pool_size, padding=padding,
                           bias=False, drop_rate=self.drop_rate))
            down_samp_len = (down_samp_len + 2 * padding - self.kernel_size) // self.pool_size + 1
            encode_len.append(down_samp_len)

        encode_len = encode_len[::-1][1:]

        for ii, depth in enumerate(range(self.depths - 2, -1, -1)):
            in_channels = num_filters
            num_filters = int(2**depth * self.filters_root)
            padding = get_padding(self.kernel_size, self.pool_size)
            output_padding = get_output_padding(down_samp_len, encode_len[ii],
                                                self.kernel_size, self.pool_size, padding)
            self.upconv_layers.append(
                ConvBNReLU(in_channels, num_filters, self.kernel_size,
                           stride=self.pool_size, padding=padding,
                           transpose=True, output_padding=output_padding,
                           bias=False, drop_rate=self.drop_rate))
            down_samp_len = encode_len[ii]

            padding = get_padding(self.kernel_size, 1)
            self.decoder_layers.append(
                ConvBNReLU(in_channels, num_filters, self.kernel_size,
                           padding=padding, bias=False, drop_rate=self.drop_rate))

        self.output_conv = nn.Conv1d(in_channels=self.filters_root,
                                     out_channels=self.n_class,
                                     kernel_size=1)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def crop_and_concat(skip, x):
        if skip.size(-1) > x.size(-1):
            d = skip.size(-1) - x.size(-1)
            skip = skip[:, :, d // 2:d // 2 + x.size(-1)]
        elif x.size(-1) > skip.size(-1):
            d = x.size(-1) - skip.size(-1)
            x = x[:, :, d // 2:d // 2 + skip.size(-1)]
        return torch.cat([skip, x], dim=1)

    def forward(self, x):
        skip_connections = []
        x = self.input_layers(x)
        for depth in range(self.depths):
            x = self.encoder_layers[depth](x)
            if depth == self.depths - 1:
                continue
            skip_connections.append(x)
            x = self.pool_layers[depth](x)

        for ii in range(self.depths - 1):
            x = self.upconv_layers[ii](x)
            x = self.crop_and_concat(skip_connections.pop(), x)
            x = self.decoder_layers[ii](x)

        return self.output_conv(x)