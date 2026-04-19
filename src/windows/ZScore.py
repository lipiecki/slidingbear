import numpy

class ZScore():
    def __init__(self, num_vars: int):
        self.loc = numpy.full(num_vars, 0.0)
        self.scale = numpy.full(num_vars, 1.0)

    def forward(self, x, var: int):
        return (x-self.loc[var])/self.scale[var]
    
    def backward(self, z, var: int):
        return z*self.scale[var] + self.loc[var]

    def backward_scale(self, z, var: int):
        return z*self.scale[var]

    def set_loc(self, loc, var: int):
        self.loc[var] = loc
    
    def set_scale(self, scale, var: int):
        self.scale[var] = scale

    def unset(self, var: int):
        self.loc[var] = 0.0
        self.scale[var] = 1.0
