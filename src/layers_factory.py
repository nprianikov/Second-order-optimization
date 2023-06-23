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
                'convIn_16': (nn.Conv2d, (self.input_shape, 16, 3, 1)),
                'convIn_32': (nn.Conv2d, (self.input_shape, 32, 3, 1)),
                'conv16_16': (nn.Conv2d, (16, 16, 3, 1)),
                'conv16_32': (nn.Conv2d, (16, 32, 3, 1)),
                'conv32_32': (nn.Conv2d, (32, 32, 3, 1)),
                'conv32_64': (nn.Conv2d, (32, 64, 3, 1)),
                'conv64_64': (nn.Conv2d, (64, 64, 3, 1)),
                'conv64_32': (nn.Conv2d, (64, 32, 3, 1)),
                'pool': (nn.MaxPool2d, (2, 1)),
                'flatten': (nn.Flatten, ()),
                'dropout': (nn.Dropout, (self.p,)),
                'fcSmall': (nn.Linear, ((23 * 23 * 32 if self.dataset == "cifar10" else 19 * 19 * 32), 32)),
                'fcDepth': (nn.Linear, ((17 * 17 * 32 if self.dataset == "cifar10" else 13 * 13 * 32), 32)),
                'fcWidth': (nn.Linear, ((23 * 23 * 64 if self.dataset == "cifar10" else 19 * 19 * 64), 64)),
                'fcDepthWidth': (nn.Linear, ((17 * 17 * 64 if self.dataset == "cifar10" else 13 * 13 * 64), 64)),
                'fc32_Out': (nn.Linear, (32, self.output_shape)),
                'fc64_Out': (nn.Linear, (64, self.output_shape))
            }

    def create_layer(self, layer_name):
        if layer_name in self.layer_configs:
            layer_type, layer_args = self.layer_configs[layer_name]
            return layer_type(*layer_args)
        else:
            raise ValueError("Invalid layer name")

