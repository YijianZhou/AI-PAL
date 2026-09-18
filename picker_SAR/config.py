from pathlib import Path
import sys

_config_dir = Path(__file__).resolve().parent
for _directory in (_config_dir, *_config_dir.parents):
  _pal_src = _directory / "PAL_src"
  if (_pal_src / "config_ai_pal.py").exists():
    if str(_pal_src) not in sys.path:
      sys.path.insert(0, str(_pal_src))
    break
else:
  raise ImportError("PAL_src/config_ai_pal.py was not found above {}".format(__file__))
from config_ai_pal import Config as AIPALConfig

class Config(AIPALConfig):
  """SAR model, training, and inference parameters."""
  def __init__(self):
    super().__init__()
    # Model structure.
    self.rnn_hidden_size = 128
    self.rnn_num_layers = 2
    self.rnn_step_len = 0.5
    self.rnn_step_stride = 0.1
    self.rnn_num_steps = int((self.win_len - self.rnn_step_len) / self.rnn_step_stride) + 1
    self.num_att_heads = 4

    # Training.
    self.num_epochs = 20
    self.batch_size = 128
    self.neg_reduction_ratio = 1.0
    self.lr = 1e-4
    self.valid_step = 5000
    self.max_checkpoints = 20
    self.summary_step = 100

    # Continuous inference.
    self.trig_thres = 0.3
