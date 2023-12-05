import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

class ResBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, pad_mode="pad", padding=1, weight_init='HeUniform', has_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, pad_mode="pad", padding=1, weight_init='HeUniform', has_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.SequentialCell()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.SequentialCell(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, pad_mode="valid", has_bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.downsample(x)
        out = self.relu(out)
        return out

if __name__ == "__main__":
    net = ResBlock(3, 64, 1)
    print(net)
    input_data = Tensor(np.ones([1, 3, 224, 224]), ms.float32)
    output = net(input_data)
    print(output.shape)