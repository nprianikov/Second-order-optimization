import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(SmallCNN, self).__init__()
        # predefined convolutions
        self.convIn_32 = nn.Conv2d(in_channels=input_shape, out_channels=32, kernel_size=3, stride=1)
        self.conv32_32 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv32_64 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        # predefined standard layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p)
        self.activation_fn = activation_fn()
        # predefined fully connected layer
        self.fcSmall = nn.Linear(in_features=(23 * 23 * 64 if dataset == "cifar10" else 19 * 19 * 64), out_features=64)
        # output layers
        self.fc64_Out = nn.Linear(in_features=64, out_features=output_shape)

    def forward(self, x):
        x = self.pool(self.activation_fn(self.convIn_32(x)))
        x = self.pool(self.activation_fn(self.conv32_32(x)))
        x = self.pool(self.activation_fn(self.conv32_64(x)))
        x = self.dropout(self.activation_fn(self.fcSmall(self.flatten(x))))
        x = self.fc64_Out(x)
        return x


class DepthCNN(SmallCNN):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(DepthCNN, self).__init__(input_shape, output_shape, activation_fn, p, dataset)
        # predefined convolutions
        self.conv64_64 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # predefined fully connected layer
        self.fcDepth = nn.Linear(in_features=(17 * 17 * 64 if dataset == "cifar10" else 13 * 13 * 64), out_features=64)

    def forward(self, x):
        x = self.dropout(self.activation_fn(self.convIn_32(x)))
        x = self.pool(self.dropout(self.activation_fn(self.conv32_32(x))))
        x = self.dropout(self.activation_fn(self.conv32_32(x)))
        x = self.pool(self.dropout(self.activation_fn(self.conv32_32(x))))
        x = self.dropout(self.activation_fn(self.conv32_64(x)))
        x = self.pool(self.dropout(self.activation_fn(self.conv64_64(x))))
        x = self.dropout(self.activation_fn(self.fcDepth(self.flatten(x))))
        x = self.fc64_Out(x)
        return x


class WidthCNN(SmallCNN):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(WidthCNN, self).__init__(input_shape, output_shape, activation_fn, p, dataset)
        # predefined convolutions
        self.convIn_64 = nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, stride=1)
        self.conv64_64 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv64_128 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv128_128 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.conv128_64 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        # predefined fully connected layer
        self.fcWidth = nn.Linear(in_features=(23 * 23 * 128 if dataset == "cifar10" else 19 * 19 * 128),
                                 out_features=128)
        # output layers
        self.fc128_Out = nn.Linear(in_features=128, out_features=output_shape)

    def forward(self, x):
        x = self.pool(self.dropout(self.activation_fn(self.convIn_64(x))))
        x = self.pool(self.dropout(self.activation_fn(self.conv64_64(x))))
        x = self.pool(self.dropout(self.activation_fn(self.conv64_128(x))))
        x = self.dropout(self.activation_fn(self.fcWidth(self.flatten(x))))
        x = self.fc128_Out(x)
        return x


class DepthWidthCNN(DepthCNN, WidthCNN):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(DepthWidthCNN, self).__init__(input_shape, output_shape, activation_fn, p, dataset)
        # predefined fully connected layer
        self.fcDepthWidth = nn.Linear(in_features=(17 * 17 * 64 if dataset == "cifar10" else 13 * 13 * 64),
                                      out_features=128)

    def forward(self, x):
        x = self.dropout(self.activation_fn(self.convIn_64(x)))
        x = self.pool(self.dropout(self.activation_fn(self.conv64_64(x))))
        x = self.dropout(self.activation_fn(self.conv64_64(x)))
        x = self.pool(self.dropout(self.activation_fn(self.conv64_64(x))))
        x = self.dropout(self.activation_fn(self.conv64_128(x)))
        x = self.pool(self.dropout(self.activation_fn(self.conv128_64(x))))
        x = self.dropout(self.activation_fn(self.fcDepthWidth(self.flatten(x))))
        x = self.fc128_Out(x)
        return x
