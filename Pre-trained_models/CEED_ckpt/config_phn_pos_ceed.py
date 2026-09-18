from config_ai_pal import Config as AIPALConfig

class Config(AIPALConfig):
  """PhaseNet model, training, and inference parameters."""
  def __init__(self):
    super().__init__()
    # Model and label structure.
    self.label_wid = 0.5
    self.depths = 5
    self.filters_root = 16
    self.kernel_size = 7
    self.pool_size = 4
    self.drop_rate = 0.0

    # Training.
    self.num_epochs = 20
    self.batch_size = 128
    self.learning_rate = 1e-3
    self.weight_decay = 0.0
    self.valid_step = 5000
    self.max_checkpoints = 20
    self.summary_step = 100

    # Continuous inference.
    self.trig_thres = 0.3
