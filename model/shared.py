from typing import Union, Tuple
from torch import nn
import torch

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class ConvBnRelu(nn.Module):
    """
    Standard convolution block with Batch normalization and Relu activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias=False,
                 use_bn=True,
                 use_relu=True):
        super(ConvBnRelu, self).__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups,
                               bias=bias)
        self._bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self._relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x) if self._bn is not None else x
        x = self._relu(x) if self._relu is not None else x
        return x


class ResNorm(nn.Module):
    def __init__(self, channels: int, lamb=0.1, eps=1e-5):
        super(ResNorm, self).__init__()
        self._eps = torch.full((1, channels, 1, 1), eps)
        self._lambda = torch.full((1, channels, 1, 1), lamb)

    def forward(self, x):
        self._eps = self._eps.to(x.device)
        self._lambda = self._lambda.to(x.device)

        identity = x
        fi_mean = x.mean((1, 3), keepdim=True)
        fi_var = x.var((1, 3), keepdim=True)
        fin = (x - fi_mean) / (fi_var + self._eps).sqrt()
        return self._lambda * identity + fin


class AdaResNorm(nn.Module):
    def __init__(self, channels: int, grad=True, eps=1e-5):
        super(AdaResNorm, self).__init__()
        self._eps = torch.full((1, channels, 1, 1), eps)
        self._rho = nn.Parameter(torch.full((1, channels, 1, 1), 0.5)) if grad else torch.full((1, channels, 1, 1), 0.5)
        self._gamma = nn.Parameter(torch.ones(1, channels, 1, 1)) if grad else 1
        self._beta = nn.Parameter(torch.zeros(1, channels, 1, 1)) if grad else 0

    def forward(self, x):
        self._eps = self._eps.to(x.device)
        self._rho = self._rho.to(x.device)

        identity = x
        fi_mean = x.mean((1, 3), keepdim=True)
        fi_var = x.var((1, 3), keepdim=True)
        fin = (x - fi_mean) / (fi_var + self._eps).sqrt()
        return self._gamma * (self._rho * identity + (1 - self._rho) * fin) + self._beta


class SubSpectralNorm(nn.Module):
    def __init__(self, channels: int, sub_bands: int):
        super(SubSpectralNorm, self).__init__()
        self._sub_bands = sub_bands
        self._bn = nn.BatchNorm2d(channels * sub_bands)

    def forward(self, x):
        # x features with shape {n, c, f, t}
        n, c, f, t = x.data.size()
        x = x.view(n, c * self._sub_bands, f // self._sub_bands, t)
        x = self._bn(x)
        x = x.view(n, c, f, t)
        return x


class BroadcastBlock(nn.Module):
    """Implementation of Broadcasted Residual Learning."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout_rate: float, sub_bands: int):
        super(BroadcastBlock, self).__init__()
        self.trans_conv = ConvBnRelu(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.freq_dw_conv = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=((kernel_size - 1) // 2, 0),
                                      groups=out_channels, bias=False)
        self.ssn = SubSpectralNorm(out_channels, sub_bands)
        self.temp_dw_conv = ConvBnRelu(out_channels, out_channels, (1, kernel_size),
                                       padding=(0, (kernel_size - 1) // 2),
                                       groups=out_channels, use_relu=False)
        self.swish = nn.SiLU()
        self.pw_conv = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.se_block = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Residual identity
        identity = x
        # Expand or shrink channels if in_channels != out_channels
        x = self.trans_conv(x) if self.trans_conv is not None else x
        # Frequency-wise convolution
        x = self.freq_dw_conv(x)
        x = self.ssn(x)
        # Auxiliary identity
        auxiliary = x
        # frequency average pooling
        x = x.mean(2, keepdim=True)
        # Temporal-wise convolution
        x = self.temp_dw_conv(x)
        x = self.swish(x)
        # Point-wise convolution
        x = self.pw_conv(x)
        x = self.dropout_layer(x)
        # Add shortcuts
        x = auxiliary + x if self.trans_conv is not None else identity + auxiliary + x

        x = self.se_block(x)

        x = self.relu(x)
        return x


class ShuffleLayer(nn.Module):
    def __init__(self, group: int):
        super(ShuffleLayer, self).__init__()
        self._group = group

    def forward(self, x):
        b, c, f, t = x.data.size()
        # assert c % self._group == 0
        group_channels = c // self._group

        x = x.reshape(b, group_channels, self._group, f, t)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, f, t)
        return x


class TimeFreqSepConvolutions(nn.Module):
    """Implementation of Time-Frequency Separable Convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout_rate: float):
        super(TimeFreqSepConvolutions, self).__init__()
        assert out_channels % 2 == 0, "Out channels must be divisible by 2"
        half_channels = out_channels // 2

        self.trans_conv = ConvBnRelu(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.freq_dw_conv = ConvBnRelu(half_channels, half_channels, (kernel_size, 1),
                                       padding=((kernel_size - 1) // 2, 0),
                                       groups=half_channels)
        self.freq_pw_conv = ConvBnRelu(half_channels, half_channels, 1)
        self.temp_dw_conv = ConvBnRelu(half_channels, half_channels, (1, kernel_size),
                                       padding=(0, (kernel_size - 1) // 2),
                                       groups=half_channels)
        self.temp_pw_conv = ConvBnRelu(half_channels, half_channels, 1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.shuffle_layer = ShuffleLayer(group=half_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        # Expand or shrink channels if in_channels != out_channels
        x = self.trans_conv(x) if self.trans_conv is not None else x
        # Channel shuffle
        x = self.shuffle_layer(x)
        # Split feature maps with half the channels
        x1, x2 = torch.split(x, x.data.size(1) // 2, dim=1)
        # Copy x1, x2 for residual path
        identity1 = x1
        identity2 = x2
        # Frequency-wise convolution block
        x1 = self.freq_dw_conv(x1)
        x1 = x1.mean(2, keepdim=True)  # frequency average pooling
        x1 = self.freq_pw_conv(x1)
        x1 = self.dropout_layer(x1)
        x1 = x1 + identity1
        # Time-wise convolution block
        x2 = self.temp_dw_conv(x2)
        x2 = x2.mean(3, keepdim=True)  # temporal average pooling
        x2 = self.temp_pw_conv(x2)
        x2 = self.dropout_layer(x2)
        x2 = x2 + identity2
        # Concat x1 and x2
        x = torch.cat((x1, x2), dim=1)
        x = self.se(x)
        return x