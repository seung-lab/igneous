import enum
from typing import Any, Dict, Tuple, Union, Optional

ShapeType = Tuple[int, int, int]

class DownsampleMethods(enum.IntEnum):
  AVERAGE_POOLING = 1
  MODE_POOLING = 2
  MIN_POOLING = 3
  MAX_POOLING = 4
  STRIDING = 5
  AUTO = 6