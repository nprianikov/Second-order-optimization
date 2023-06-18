import torch
from torch import nn
from src.layers_factory import LayersFactory


class SmallCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(SmallCNN, self).__init__()
        # factory
        self.lfc = LayersFactory(input_shape, output_shape, p, dataset)
        # unique
        self.convIn_32 = self.lfc.create_layer('convIn_32')
        self.pool = self.lfc.create_layer('pool')
        self.dropout = self.lfc.create_layer('dropout')
        self.activation_fn = activation_fn()

        self.conv1 = nn.Sequential(
            self.convIn_32,
            self.activation_fn,
            self.pool,
        )

        self.conv2 = self.lfc.create_layer('conv32_32')
            
        self.conv3 = self.lfc.create_layer('conv32_64')

        self.flatten = self.lfc.create_layer('flatten')
        self.fcSmall = self.lfc.create_layer('fcSmall')
        self.fc64_Out = self.lfc.create_layer('fc64_Out')

        self.out = nn.Sequential(
            self.flatten,
            self.fcSmall,
            self.activation_fn,
            self.dropout,
            self.fc64_Out,
        )
        
    def forward(self, x):
        """
        Forward pass through model
        Returns:
        - x: outputs scores from final layer
        - a: outputs of layers (pre-activation)
        - h: inputs of layers
        """
        # TODO: append to h before or after pooling layer? + flattening

        a = [] 
        h = []

        # layer 1
        h.append(x)
        x = self.convIn_32(x)
        a.append(x)
        x = self.activation_fn(x)
        # layer 2
        h.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        a.append(x)
        x = self.activation_fn(x)
        # layer 3
        h.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        a.append(x)
        x = self.activation_fn(x)
        # layer 4
        h.append(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fcSmall(x)
        a.append(x)
        x = self.activation_fn(x)
        # layer 5
        h.append(x)
        x = self.dropout(x)
        x = self.fc64_Out(x)
        a.append(x)

        return x, a, h


class DepthCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(DepthCNN, self).__init__()
        # factory
        self.lfc = LayersFactory(input_shape, output_shape, p, dataset)
        # unique
        self.convIn_32 = self.lfc.create_layer('convIn_32')
        self.activation_fn = activation_fn()
        self.pool = self.lfc.create_layer('pool')
        self.dropout = self.lfc.create_layer('dropout')

        self.conv1 = nn.Sequential(
            self.convIn_32,
            self.activation_fn,
            self.dropout,
        )

        self.conv2 = nn.Sequential(
            self.lfc.create_layer('conv32_32'),
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.conv3 = nn.Sequential(
            self.lfc.create_layer('conv32_32'),
            self.dropout,
            self.activation_fn,
        )

        self.conv4 = nn.Sequential(
            self.lfc.create_layer('conv32_32'),
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.conv5 = nn.Sequential(
            self.lfc.create_layer('conv32_64'),
            self.activation_fn,
            self.dropout,
        )

        self.conv6 = nn.Sequential(
            self.lfc.create_layer('conv64_64'),
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.flatten = self.lfc.create_layer('flatten')
        self.fcDepth = self.lfc.create_layer('fcDepth')
        self.fc64_Out = self.lfc.create_layer('fc64_Out')

        self.out = nn.Sequential(
            self.flatten,
            self.fcDepth,
            self.activation_fn,
            self.dropout,
            self.fc64_Out,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.out(x)
        return x


class WidthCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(WidthCNN, self).__init__()
        # factory
        self.lfc = LayersFactory(input_shape, output_shape, p, dataset)
        # unique
        self.convIn_64 = self.lfc.create_layer('convIn_64')
        self.activation_fn = activation_fn()
        self.pool = self.lfc.create_layer('pool')
        self.dropout = self.lfc.create_layer('dropout')

        self.conv1 = nn.Sequential(
            self.convIn_64,
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.conv2 = nn.Sequential(
            self.lfc.create_layer('conv64_64'),
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.conv3 = nn.Sequential(
            self.lfc.create_layer('conv64_128'),
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.flatten = self.lfc.create_layer('flatten')
        self.fcWidth = self.lfc.create_layer('fcWidth')
        self.fc128_Out = self.lfc.create_layer('fc128_Out')

        self.out = nn.Sequential(
            self.flatten,
            self.fcWidth,
            self.activation_fn,
            self.dropout,
            self.fc128_Out,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out(x)
        return x


class DepthWidthCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(DepthWidthCNN, self).__init__()
        # factory
        self.lfc = LayersFactory(input_shape, output_shape, p, dataset)
        # unique
        self.convIn_64 = self.lfc.create_layer('convIn_64')
        self.activation_fn = activation_fn()
        self.pool = self.lfc.create_layer('pool')
        self.dropout = self.lfc.create_layer('dropout')

        self.conv1 = nn.Sequential(
            self.convIn_64,
            self.activation_fn,
            self.dropout,
        )

        self.conv2 = nn.Sequential(
            self.lfc.create_layer('conv64_64'),
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.conv3 = nn.Sequential(
            self.lfc.create_layer('conv64_64'),
            self.activation_fn,
            self.dropout,
        )

        self.conv4 = nn.Sequential(
            self.lfc.create_layer('conv64_64'),
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.conv5 = nn.Sequential(
            self.lfc.create_layer('conv64_128'),
            self.activation_fn,
            self.dropout,
        )

        self.conv6 = nn.Sequential(
            self.lfc.create_layer('conv128_128'),
            self.activation_fn,
            self.dropout,
            self.pool,
        )

        self.flatten = self.lfc.create_layer('flatten')
        self.fcDepthWidth = self.lfc.create_layer('fcDepthWidth')
        self.fc128_Out = self.lfc.create_layer('fc128_Out')

        self.out = nn.Sequential(
            self.flatten,
            self.fcDepthWidth,
            self.activation_fn,
            self.dropout,
            self.fc128_Out,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.out(x)
        return x
