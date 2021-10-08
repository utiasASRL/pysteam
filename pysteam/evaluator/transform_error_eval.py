from typing import Optional
import numpy as np

from pylgmath import Transformation
from ..state import TransformStateVar
from . import Evaluator, TransformEvaluator, ComposeTransformEvaluator, InverseTransformEvaluator, LogMapEvaluator, FixedTransformEvaluator, TransformStateEvaluator


class TransformErrorEval(Evaluator):
  """Transformation error function evaluator"""

  def __init__(self,
               *,
               T: Optional[TransformEvaluator] = None,
               meas_T_21: Optional[Transformation] = None,
               T_21: Optional[TransformEvaluator] = None,
               T_20: Optional[TransformStateVar] = None,
               T_10: Optional[TransformStateVar] = None) -> None:
    super().__init__()

    # error is difference between 'T' and identity (in Lie algebra space)
    if T is not None:
      self._error_eval: TransformEvaluator = LogMapEvaluator(T)
      return

    assert meas_T_21 is not None
    meas = FixedTransformEvaluator(meas_T_21)

    # error between meas_T_21 and T_21
    if T_21 is not None:
      self._error_eval: TransformEvaluator = LogMapEvaluator(
          ComposeTransformEvaluator(meas, InverseTransformEvaluator(T_21)))
      return

    # error between meas_T_21 and T_20*inv(T_10)
    if T_20 is not None and T_10 is not None:
      T_10 = TransformStateEvaluator(T_10)
      T_20 = TransformStateEvaluator(T_20)
      self._error_eval: TransformEvaluator = LogMapEvaluator(
          ComposeTransformEvaluator(ComposeTransformEvaluator(meas, T_10), InverseTransformEvaluator(T_20)))
      return

    raise RuntimeError("Unknown initialization.")

  def is_active(self):
    return self._error_eval.is_active()

  def evaluate(self, lhs: Optional[np.ndarray] = None):
    return self._error_eval.evaluate(lhs)