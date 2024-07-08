import torch.nn as nn

class PoseDecoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.nl = nn.ReLU()
        self.squeeze = nn.Conv2d(input_channels, 256, 1)  #创建一个卷积层实例，该层将输入的通道数压缩为256，使用1x1的卷积核。
        self.conv_1 = nn.Conv2d(256, 256, 3, 1, 1)  #创建第一个卷积层实例，该层输入通道和输出通道都是256，使用3x3的卷积核，步长为1，填充为1
        self.conv_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv_3 = nn.Conv2d(256, 6, 1)  #该层将输入通道压缩为6，使用1x1的卷积核


        self.conv_3.weight.data *= 0.01
        self.conv_3.bias.data *= 0.01  #第三个卷积层的权重和偏置进行初始化，将它们的值乘以0.01。

    def forward(self, x):
        x = self.squeeze(x)
        x = self.nl(x)

        x = self.conv_1(x)
        x = self.nl(x)

        x = self.conv_2(x)
        x = self.nl(x)

        x = self.conv_3(x)
        x = x.mean((3, 2)).view(-1, 1, 1, 6)

        x_angle = x[..., :3]
        x_translation = x[..., 3:]

        return x_angle, x_translation
