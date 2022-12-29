from abc import ABC, abstractmethod
from typing import Optional

from dto.plane import Plane


class AssocMethod(ABC):
    @abstractmethod
    def get_result(self, prev: Plane, cur: Plane) -> Optional[float]:
        pass
