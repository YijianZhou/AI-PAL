from config_ai_pal import Config as AIPALConfig

class Config(AIPALConfig):
  """Frame Transformer model, training, and inference parameters."""
  def __init__(self):
    super().__init__()
    # Model structure. Class order: Noise, P, S.
    self.ft_frame_length = 0.5
    self.ft_frame_step = 0.1
    self.ft_d_model = 256
    self.ft_num_heads = 4
    self.ft_num_layers = 5
    self.ft_ffn_hidden = 512
    self.ft_norm_eps = 1e-6
    self.ft_rotary_dim = 64
    self.ft_max_sequence_length = 512
    self.ft_rope_base_theta = 10000.0
    self.ft_embedding_dropout = 0.05
    self.ft_attention_dropout = 0.10
    self.ft_output_dropout = 0.05
    self.ft_ffn_dropout = 0.10

    # Training.
    self.num_epochs = 20
    self.batch_size = 128
    self.neg_reduction_ratio = 1.0
    self.learning_rate = 1e-4
    self.min_learning_rate = 1e-6
    self.warmup_steps = 10000
    self.weight_decay = 1e-4
    self.adam_betas = (0.9, 0.999)
    self.adam_eps = 1e-8
    self.amp = True
    self.grad_clip_norm = 1.0
    self.valid_step = 5000
    self.max_checkpoints = 20
    self.summary_step = 100

    # Continuous inference.
    self.trig_thres = 0.3
