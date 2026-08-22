""" Defination of Self-Attention RNN (SAR) Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from . import config
except ImportError:
    import config
cfg = config.Config()

class SAR(nn.Module):
  def __init__(self):
    super(SAR, self).__init__()
    # hyper-params
    self.input_size = int(cfg.rnn_step_len * cfg.num_chn * cfg.samp_rate)
    self.hidden_size = cfg.rnn_hidden_size
    self.num_layers = cfg.rnn_num_layers
    self.num_heads = cfg.num_att_heads
    # def layers
    self.gru_layer = nn.GRU(input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        bidirectional=True,
        batch_first=True)
    if self.num_heads>0:
        self.attention = nn.MultiheadAttention(embed_dim=2*self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True)
    else: self.attention = None
    self.fc_layer = nn.Linear(2*self.hidden_size, 3)

  def forward(self, x):
    x, _ = self.gru_layer(x)
    if self.attention: x, _ = self.attention(query=x, key=x, value=x)  # self attention
    return self.fc_layer(x)
