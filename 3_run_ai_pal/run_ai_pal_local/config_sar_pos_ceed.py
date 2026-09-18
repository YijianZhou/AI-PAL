from config_ai_pal import Config as AIPALConfig


class Config(AIPALConfig):
  """SAR positive-picker model, training, and inference parameters."""
  def __init__(self):
    super().__init__()
    # Model structure.
    self.rnn_hidden_size = 128
    self.rnn_num_layers = 2
    self.rnn_step_len = 0.5
    self.rnn_step_stride = 0.1
    self.rnn_num_steps = int(
      (self.win_len - self.rnn_step_len) / self.rnn_step_stride
    ) + 1
    self.num_att_heads = 4

    # Training.
    self.num_epochs = 20
    self.batch_size = 128
    self.lr = 1e-4
    self.valid_step = 5000
    self.summary_step = 100

    # Positive-event inference.
    self.trig_thres = 0.3
