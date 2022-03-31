import abc
from typing import List

from ..problem import CostTerm


class Interface(abc.ABC):
  """Base class of the trajectory interface"""

  @abc.abstractmethod
  def get_prior_cost_terms() -> List[CostTerm]:
    """Trajectory prior cost terms to be added to the optimization problem"""