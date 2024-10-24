from mytorch.tensor import Tensor, Dependency
from mytorch.model import Model

class Mymodel(Model):
    def __init__(self):
        pass

    def __forward__(self, x: Tensor):
        ...
        return x
    
