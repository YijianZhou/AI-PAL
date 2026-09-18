from config_ai_pal import Config as AIPALConfig

class Config(AIPALConfig):
  """Res-U-Net model, training, and inference parameters."""
  def __init__(self):
    super().__init__()
    # Model and label structure.
    self.label_wid = 0.5
    self.run_drop_rate = 0.0
    self.run_stem_channels = 16
    self.run_stem_kernel = 15
    self.run_stem_padding = 7
    self.run_stage_channels = [16, 32, 64, 128, 256]
    self.run_stage_kernels = [9, 9, 7, 5, 3]
    self.run_stage_blocks = [1, 1, 1, 1, 1]
    self.run_down_channels = [32, 64, 128, 256, 256]
    self.run_down_kernel = 5
    self.run_down_stride = 2
    self.run_down_padding = 2
    self.run_bottleneck_channels = 256
    self.run_bottleneck_kernel = 3
    self.run_bottleneck_blocks = 1
    self.run_up_channels = [256, 128, 64, 32, 16]
    self.run_up_kernel = 5
    self.run_up_stride = 2
    self.run_up_padding = 2
    self.run_up_output_padding = [0, 0, 0, 1, 1]
    self.run_decoder_kernels = [3, 5, 7, 9, 9]
    self.run_decoder_blocks = [1, 1, 1, 1, 1]
    self.run_output_kernel = 1

    # Training.
    self.num_epochs = 20
    self.batch_size = 128
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
