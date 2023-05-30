import torch
from torch import nn

class LayersFactory():
    def __init__(self, input_shape, output_shape, p, dataset) -> None:
        super(LayersFactory, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.p = p
        self.dataset = dataset

        self.layer_configs = {
                'convIn_32': (nn.Conv2d, (self.input_shape, 32, 3, 1)),
                'convIn_64': (nn.Conv2d, (self.input_shape, 64, 3, 1)),
                'conv32_32': (nn.Conv2d, (32, 32, 3, 1)),
                'conv32_64': (nn.Conv2d, (32, 64, 3, 1)),
                'conv64_64': (nn.Conv2d, (64, 64, 3, 1)),
                'conv64_128': (nn.Conv2d, (64, 128, 3, 1)),
                'conv128_128': (nn.Conv2d, (128, 128, 3, 1)),
                'conv128_64': (nn.Conv2d, (128, 64, 3, 1)),
                'pool': (nn.MaxPool2d, (2, 1)),
                'flatten': (nn.Flatten, ()),
                'dropout': (nn.Dropout, (self.p,)),
                'fcSmall': (nn.Linear, ((23 * 23 * 64 if self.dataset == "cifar10" else 19 * 19 * 64), 64)),
                'fcDepth': (nn.Linear, ((17 * 17 * 64 if self.dataset == "cifar10" else 13 * 13 * 64), 64)),
                'fcWidth': (nn.Linear, ((23 * 23 * 128 if self.dataset == "cifar10" else 19 * 19 * 128), 128)),
                'fcDepthWidth': (nn.Linear, ((17 * 17 * 128 if self.dataset == "cifar10" else 13 * 13 * 128), 128)),
                'fc64_Out': (nn.Linear, (64, self.output_shape)),
                'fc128_Out': (nn.Linear, (128, self.output_shape))
            }

    def create_layer(self, layer_name):
        if layer_name in self.layer_configs:
            layer_type, layer_args = self.layer_configs[layer_name]
            return layer_type(*layer_args)
        else:
            raise ValueError("Invalid layer name")

