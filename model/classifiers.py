from torch import nn


class ConvClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(ConvClassifier, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean((-1, -2), keepdim=False)
        return x


class SingleLinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(SingleLinearClassifier, self).__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = x.mean(dim=1)
        return x


class MultiLayerPerception(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, num_classes: int, dropout: float):
        super(MultiLayerPerception, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_units)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x