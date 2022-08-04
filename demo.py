import torch
import torch.nn as nn
import torch.functional as F

class SiLU(nn.Module):      # 其实pytorch1.7以后的版本集成了silu激活函数，可以直接通过nn.SiLU调用，但这里我们按原版代码
    @staticmethod           # 静态方法，在此之后可以通过类名调用该函数
    def forward(x):
        return x * torch.sigmoid(x)


class MyModel(nn.Module):
    def __init__(self, inchannels, outchannels, ksize, stride, groups=1, bias=False, act='silu'):
        super().__init__()
        pad = (ksize - 1) // 2
        self.myconv = nn.Conv2d(in_channels=inchannels,
                                out_channels=outchannels,
                                kernel_size=ksize,
                                stride=stride,
                                padding=pad,
                                groups=groups,
                                bias=bias
                                )
        self.bn = nn.BatchNorm2d(outchannels, eps=1e-6, momentum=0.3, affine=True, track_running_stats=True)
        self.act = SiLU()

    def forward(self, x):
        return self.act(self.bn(self.myconv(x)))


net = MyModel(inchannels=3, outchannels=5, ksize=3, stride=1, groups=1, bias=True)
torch.manual_seed(21)
x = torch.randn(1, 3, 24, 24)
print(x)
print('*' * 100)
y = net(x)
print(y)


