import numpy
from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def forward(self, x, var: int):
        pass

    @abstractmethod
    def backward(self, z, var: int):
        pass

    @abstractmethod
    def backward_scale(self, z, var: int):
        pass

    def set_loc(self, loc, var: int):
        self.loc[var] = loc

    def set_scale(self, scale, var: int):
        scale = max(scale, 1e-6)
        self.scale[var] = scale

    def unset(self, var: int):
        self.loc[var] = 0.0
        self.scale[var] = 1.0


class ZScore(Transform):
    def __init__(self, num_vars: int):
        self.loc = numpy.full(num_vars, 0.0)
        self.scale = numpy.full(num_vars, 1.0)

    def forward(self, x, var: int):
        return (x - self.loc[var]) / self.scale[var]

    def backward(self, z, var: int):
        return z * self.scale[var] + self.loc[var]

    def backward_scale(self, z, var: int):
        return z * self.scale[var]


class Asinh(Transform):
    def __init__(self, num_vars: int):
        self.loc = numpy.full(num_vars, 0.0)
        self.scale = numpy.full(num_vars, 1.0)

    def forward(self, x, var: int):
        z = (x - self.loc[var]) / self.scale[var]
        return numpy.arcsinh(z)

    def backward(self, z, var: int):
        return numpy.sinh(z) * self.scale[var] + self.loc[var]

    def backward_scale(self, z, var: int):
        return numpy.sinh(z) * self.scale[var]
    
class BoxCox(Transform):
    def __init__(self, num_vars: int, param: float = 0.5):
        self.loc = numpy.full(num_vars, 0.0)
        self.scale = numpy.full(num_vars, 1.0)
        self.param = param

    def forward(self, x, var: int):
        z = (x - self.loc[var]) / self.scale[var]
        return numpy.sign(z)*numpy.add(numpy.power(numpy.add(numpy.abs(z), 1), self.param), -1)/self.param

    def backward(self, z, var: int):
        return numpy.sign(z)*numpy.add(numpy.power(numpy.add(numpy.abs(z)*self.param, 1), 1/self.param), -1) * self.scale[var] + self.loc[var]

    def backward_scale(self, z, var: int):
        return numpy.sign(z)*numpy.add(numpy.power(numpy.add(numpy.abs(z)*self.param, 1), 1/self.param), -1) * self.scale[var]

def match_transform(transform_name: str):
    match transform_name:
        case "zscore":
            return ZScore
        case "asinh":
            return Asinh
        case "boxcox":
            return BoxCox
        case _:
            return None
