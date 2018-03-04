import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    def __init__(self, hidden_channels = 10):
        super(SimpleConvNet, self).__init__()
        
        self.toroidal_boundary_padding_size = 2

        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=3)

    def forward(self, x):
        # two 3x3 kernels means 2 layers will be taken from all sides
        hidden = F.relu(self.conv1(x))
        return F.relu(self.conv2(hidden))