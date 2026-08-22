from config_ai_pal import Config as AIPALConfig


class Config(AIPALConfig):
  """SeisBench PhaseNet reference-picker inference parameters."""
  def __init__(self):
    super().__init__()
    self.weights = "original"
    # Original PhaseNet uses a 3000-sample input. Keep this product fixed if
    # samp_rate changes; e.g., 30 s * 100 Hz = 3000 samples.
    self.win_len = 30.0
    self.overlap_sec = 15.0
    self.trig_thres = 0.3
