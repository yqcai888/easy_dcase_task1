import torch
import torch.nn as nn
from model.beats.BEATs import BEATsConfig, BEATs
from model.classifiers import SingleLinearClassifier, ConvClassifier
from model.shared import ConvBnRelu, ResNorm, AdaResNorm, BroadcastBlock, TimeFreqSepConvolutions


class DCASEBaselineCnn3(nn.Module):
    """
    Previous baseline system of the task 1 of DCASE Challenge.
    A simple CNN consists of 3 conv layers and 2 linear layers.

    Note: Kernel size of the Max-pooling layers need to be change for adapting to different size of inputs.

    Args:
        in_channels (int): Number of input channels. (default: ``1``)
        num_classes (int): Number of output classes. (default: ``10``)
        base_channels (int): Number of base channels. (default: ``32``).
        kernel_size (int): Kernel size of convolution layers. (default: ``7``).
        dropout (float): Dropout rate. (default: ``0.3``).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 16, kernel_size: int = 7,
                 dropout: float = 0.3):
        super(DCASEBaselineCnn3, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=True)
        self.conv2 = ConvBnRelu(base_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=True)
        self.conv3 = ConvBnRelu(base_channels, base_channels * 2, kernel_size, padding=(kernel_size - 1) // 2,
                                bias=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        # Adjust the kernel of max_pooling util the output shape is (F=2, T=1)
        self.max_pooling1 = nn.MaxPool2d((5, 5))
        self.max_pooling2 = nn.MaxPool2d((4, 10))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(base_channels * 4, 100)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(100, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # print('INPUT SHAPE:', x.shape)
        x = self.conv1(x)

        # print('CONV2 INPUT SHAPE:', x.shape)
        x = self.conv2(x)
        x = self.max_pooling1(x)
        x = self.dropout1(x)

        # print('CONV3 INPUT SHAPE:', x.shape)
        x = self.conv3(x)
        x = self.max_pooling2(x)
        x = self.dropout2(x)

        # print('FLATTEN INPUT SHAPE:', x.shape)
        x = self.flatten(x)

        # print('LINEAR INPUT SHAPE:', x.shape)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        # print('OUTPUT SHAPE:', x.shape)
        return x


class BcResNet(nn.Module):
    """
    Implementation of BC-ResNet, based on Broadcasted Residual Learning.
    Check more details at: https://arxiv.org/abs/2106.04140

    Args:
        in_channels (int): Number of input channels. (default: ``1``)
        num_classes (int): Number of output classes. (default: ``10``)
        base_channels (int): Number of base channels that controls the complexity of model. (default: ``40``)
        kernel_size (int): Kernel size of each convolutional layer in BC blocks. (default: ``3``)
        dropout (float): Dropout rate. (default: ``0.1``)
        sub_bands (int): Number of sub-bands for SubSpectralNorm (SSN). ``1`` indicates SSN is not applied. (default: ``1``)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 40, depth: int = 15,
                 kernel_size: int = 3, dropout: float = 0.1, sub_bands: int = 1):
        super(BcResNet, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.sub_bands = sub_bands

        cfg = {
            15: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Conv2d(in_channels, 2 * base_channels, 5, stride=2, bias=False, padding=2)
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = 2 * width
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [BroadcastBlock(vt, v, self.kernel_size, self.dropout, self.sub_bands)]
                vt = v
            else:
                layers += [BroadcastBlock(vt, vt, self.kernel_size, self.dropout, self.sub_bands)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x


class TFSepNet(nn.Module):
    """
    Implementation of TF-SepNet-64, based on Time-Frequency Separate Convolutions. Check more details at:
    https://ieeexplore.ieee.org/abstract/document/10447999 and
    https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Cai_61_t1.pdf

    Args:
        in_channels (int): Number of input channels. (default: ``1``)
        num_classes (int): Number of output classes. (default: ``10``)
        base_channels (int): Number of base channels that controls the complexity of model. (default: ``64``)
        kernel_size (int): Kernel size of each convolutional layer in TF-SepConvs blocks. (default: ``3``)
        dropout (float): Dropout rate. (default: ``0.1``)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, base_channels: int = 64, depth: int = 17,
                 kernel_size: int = 3, dropout: float = 0.1):
        super(TFSepNet, self).__init__()
        assert base_channels % 2 == 0, "Base_channels should be divisible by 2."
        self.dropout = dropout
        self.kernel_size = kernel_size

        # Two settings of the depth. ``17`` have an additional Max-pooling layer before the final block of TF-SepConvs.
        cfg = {
            16: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 2.5, 2.5, 2.5, 'N'],
            17: ['N', 1, 1, 'N', 'M', 1.5, 1.5, 'N', 'M', 2, 2, 'N', 'M', 2.5, 2.5, 2.5, 'N'],
        }

        self.conv_layers = nn.Sequential(ConvBnRelu(in_channels, base_channels // 2, 3, stride=2, padding=1),
                                         ConvBnRelu(base_channels // 2, 2 * base_channels, 3, stride=2, padding=1,
                                                    groups=base_channels // 2))
        # Compute the number of channels for each layer.
        layer_config = [int(i * base_channels) if not isinstance(i, str) else i for i in cfg[depth]]
        self.middle_layers = self._make_layers(base_channels, layer_config)
        # Get the index of channel number for the cla_layer.
        last_num_index = -1 if not isinstance(layer_config[-1], str) else -2
        # 1x1 convolution layer as the cla_layer.
        self.classifier = ConvClassifier(layer_config[last_num_index], num_classes)

    def _make_layers(self, width: int, layer_config: list):
        layers = []
        vt = width * 2
        for v in layer_config:
            if v == 'N':
                layers += [ResNorm(channels=vt)]
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v != vt:
                layers += [TimeFreqSepConvolutions(vt, v, self.kernel_size, self.dropout)]
                vt = v
            else:
                layers += [TimeFreqSepConvolutions(vt, vt, self.kernel_size, self.dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.middle_layers(x)
        x = self.classifier(x)
        return x


class PretrainedBEATs(nn.Module):
    """
    Module wrapping a BEATs encoder with pretrained weights and a new linear classifier.
    Check more details at: https://arxiv.org/abs/2212.09058

    Args:
        pretrained (str): path to the pretrained checkpoint. Leave ``None`` when no need pretrained.
    """

    def __init__(self, pretrained=None, num_classes=10, **kwargs):
        super(PretrainedBEATs, self).__init__()
        # Load model config and weights from checkpoints when use pretrained, otherwise use default settings
        ckpt = torch.load(pretrained) if pretrained else None
        hyperparams = ckpt['cfg'] if pretrained else kwargs
        cfg = BEATsConfig(hyperparams)
        self.encoder = BEATs(cfg)
        if pretrained:
            self.encoder.load_state_dict(ckpt['model'], strict=False)
        # Create a new linear classifier
        self.classifier = SingleLinearClassifier(in_features=cfg.encoder_embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = self.encoder.extract_features(x)[0]
        return self.classifier(x)
