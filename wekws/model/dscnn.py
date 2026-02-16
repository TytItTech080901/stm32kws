import torch
import torch.nn as nn
import torch.nn.functional as F


class DSCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        # layers: int,
        out_dim: int,
    ):
        """
        Args:
            input_dim: The dimension of the input data.
            layers: The number of layers in the model.
            kernel_size: The size of the convolutional kernel.
            num_classes: The number of output classes.
            stride: The stride for convolutional layers.
            padding: The padding for convolutional layers.
            out_dim: The dimension of the output features.
        """

        self.input_dim = input_dim
        # self.layers = layers
        self.out_dim = out_dim

        super(DSCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64,
                                kernel_size=(4, 10), padding=(1, 4), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=12*5*64, out_features=out_dim)
        self.avgpool = nn.AvgPool2d(kernel_size=(1*4), stride=(1, 1), padding=(0, 0))
        self.depthwise = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(
                                    3, 3), padding=(1, 1), stride=(1, 1), groups=64)
        self.pointwise = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.pooling = nn.MaxPool2d(kernel_size=(2, 1))

    def dw_conv(self, x):
        x = self.depthwise(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = F.relu(x)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dw_conv(x)
        x = self.dw_conv(x)
        x = self.dw_conv(x)
        x = self.dw_conv(x)
        #x = self.avgpool(x)
        x = self.pooling(x)
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x
