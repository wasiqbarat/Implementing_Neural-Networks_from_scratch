from tensor import Tensor, Dependency
from layer import Linear, Conv2d, MaxPool2d
from activation import relu, softmax
import model
import numpy as np

class MyModel(model.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=4, 
                           kernel_size=(2, 2), stride=(1, 1), 
                           padding=(1, 1), need_bias=True)
        
        self.conv2 = Conv2d(in_channels=4, out_channels=8, 
                           kernel_size=(2, 2), stride=(1, 1), 
                           padding=(0, 0), need_bias=True)
        
        self.fc1 = Linear(32, 16) 
        self.fc2 = Linear(16, 3)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = Tensor(x.data.reshape(batch_size, 1, 2, 2))
        
        x = self.conv1.forward(x)
        x = relu(x)
        
        x = self.conv2.forward(x)
        x = relu(x)
    
        x = Tensor(x.data.reshape(batch_size, -1))
    
        x = self.fc1.forward(x)
        x = relu(x)
        x = self.fc2.forward(x)
        x = softmax(x)
        
        return x
