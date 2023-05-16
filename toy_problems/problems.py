from abc import ABC, abstractmethod
import torch

class Problem(ABC):
    @abstractmethod
    def objective(self, x):
        pass

    @abstractmethod
    def solutions(self):
        pass

    @abstractmethod
    def bounds(self):
        pass

class Rosenbrock(Problem):
    def __init__(self) -> None:
        super(Rosenbrock, self).__init__()

    def objective(self, x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def solutions(self):
        return [(1, 1)]
    
    def bounds(self):
        return [(-.1, 1.5), (-.1, 1.5)]
    
class Himmelblau(Problem):
    def __init__(self) -> None:
        super(Himmelblau, self).__init__()

    def objective(self, x):
        return (x[0]**2.0 + x[1] - 11.0)**2.0 + (x[0] + x[1]**2.0 - 7.0)**2.0

    def solutions(self):
        return [(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)]
    
    def bounds(self):
        return [(-5, 5), (-5, 5)]
    