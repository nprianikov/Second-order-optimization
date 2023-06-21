import torch
from torch import nn
from src.layers_factory import LayersFactory

# TODO: refactor remaining models

class SmallCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(SmallCNN, self).__init__()
        # factory
        self.lfc = LayersFactory(input_shape, output_shape, p, dataset)
        # unique
        self.pool = self.lfc.create_layer('pool')
        self.dropout = self.lfc.create_layer('dropout')
        self.activation_fn = activation_fn()
        self.kernel_size = 3

        # layers
        self._layers = []
        self._layers_names = []
        self._layers_input_size = []
        self._layers_output_size = []
        self._tau = []
        self.layers_params = []
        self.a = []
        self.h = []
        # input
        self.conv1 = self.lfc.create_layer('convIn_32')
        self._layers.append(self.conv1)
        self._layers_names.append('conv')
        # conv2
        self.conv2 = nn.Sequential(
            self.pool,
            self.lfc.create_layer('conv32_32')
        )
        self._layers.append(self.conv2)
        self._layers_names.append('conv')
        # conv3
        self.conv3 = nn.Sequential(
            self.pool,
            self.lfc.create_layer('conv32_64')
        )
        self._layers.append(self.conv3)
        self._layers_names.append('conv')
        # flatten
        self.flat = nn.Sequential(
            self.pool,
            self.lfc.create_layer('flatten'),
        )
        # fully connected
        self.fc = nn.Sequential(
            self.lfc.create_layer('fcSmall')
        )
        self._layers.append(self.fc)
        self._layers_names.append('fc')
        # out
        self.out = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('fc64_Out')
        )
        self._layers.append(self.out)
        self._layers_names.append('fc')

        # number of layers (wihout flattening)
        self.numlayers = len(self._layers)
        # accessing layer weights in the matrix form W: (out x in), b: (out x 1)
        self.layers_weights = []
        grouped = zip(*[iter(self.parameters())]*2)
        for l, (param1, param2) in enumerate(grouped):
            layers_weights_l = {}
            if self._layers_names[l] == 'conv':
                layers_weights_l['W'] = param1.reshape(param1.size()[0], -1)
                layers_weights_l['b'] = param2
                self._layers_input_size.append(layers_weights_l['W'].size()[1])
            elif self._layers_names[l] == 'fc':
                layers_weights_l['W'] = param1
                layers_weights_l['b'] = param2
                self._layers_input_size.append(param1.size()[1])

            self._layers_output_size.append(param1.size()[0])
            self.layers_weights.append(layers_weights_l)

        for l in range(len(self._layers)):
            layer_i = {}
            layer_i['name'] = self._layers_names[l]
            layer_i['input_size'] = self._layers_input_size[l]
            layer_i['output_size'] = self._layers_output_size[l]
            self.layers_params.append(layer_i)
        
    def forward(self, x):
        self.a = [] # pre-activations
        self.h = [] # layer inputs (input+post-activations)
        input_ = x
        fc_counter = 0

        for l in range(len(self._layers)):
            if self.layers_params[l]['name'] == 'fc' and fc_counter == 0:
                # flattening has no trainable parameters
                input_ = self.flat(input_)
                fc_counter += 1
            self.h.append(input_)
            a_l = self._layers[l](input_)
            input_ = self.activation_fn(a_l)
            if a_l.requires_grad:
                a_l.retain_grad() 
            self.a.append(a_l)

        if len(self._tau) == 0:
            for l in range(len(self._layers)):
                if self.layers_params[l]['name'] == 'fc':
                    self.layers_params[l]['tau'] = 1.0
                elif self.layers_params[l]['name'] == 'conv':
                    self.layers_params[l]['tau'] = self.a[l].size()[2] * self.a[l].size()[3]

        return self.a[len(self.a)-1]
    
    def get_layers_weights(self):
        self.layers_weights = []
        grouped = zip(*[iter(self.parameters())]*2)
        for l, (param1, param2) in enumerate(grouped):
            layers_weights_l = {}
            if self._layers_names[l] == 'conv':
                layers_weights_l['W'] = param1.reshape(param1.size()[0], -1)
                layers_weights_l['b'] = param2
            elif self._layers_names[l] == 'fc':
                layers_weights_l['W'] = param1
                layers_weights_l['b'] = param2
            self.layers_weights.append(layers_weights_l)
        return self.layers_weights


class DepthCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(DepthCNN, self).__init__()
        # factory
        self.lfc = LayersFactory(input_shape, output_shape, p, dataset)
        # unique
        self.activation_fn = activation_fn()
        self.pool = self.lfc.create_layer('pool')
        self.dropout = self.lfc.create_layer('dropout')
        
        # layers
        self._layers = []
        self.layers_params = []
        self.a = []
        self.h = []
        # input
        self.conv1 = self.lfc.create_layer('convIn_32')
        self._layers.append(self.conv1)
        self._layers_names.append('conv')
        # conv2
        self.conv2 = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('conv32_32'),
        )
        self._layers.append(self.conv2)
        self._layers_names.append('conv')
        # conv3
        self.conv3 = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('conv32_32'),
        )
        self._layers.append(self.conv3)
        self._layers_names.append('conv')
        # conv4
        self.conv4 = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('conv32_32'),
        )
        self._layers.append(self.conv4)
        self._layers_names.append('conv')
        # conv5
        self.conv5 = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('conv32_64'),
        )
        self._layers.append(self.conv5)
        self._layers_names.append('conv')
        # conv6
        self.conv6 = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('conv64_64'),
        )
        self._layers.append(self.conv6)
        self._layers_names.append('conv')
        # fully connected
        self.fc = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('flatten'),
            self.lfc.create_layer('fcDepth')
        )
        self._layers.append(self.fc)
        self.layers_params.append('fc')
        # out
        self.out = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('fc64_Out')
        )
        self._layers.append(self.out)
        self.layers_params.append('fc')


    def forward(self, x):
        self.a = []
        self.h = []
        input_ = x

        for l in range(len(self._layers)):
            self.h.append(input_)
            a_l = self._layers[l](input_)
            input_ = self.activation_fn(a_l)
            if a_l.requires_grad:
                a_l.retain_grad() 
            self.a.append(a_l)

        return self.a[len(self.a)-1]


class WidthCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(WidthCNN, self).__init__()
        # factory
        self.lfc = LayersFactory(input_shape, output_shape, p, dataset)
        # unique
        self.activation_fn = activation_fn()
        self.pool = self.lfc.create_layer('pool')
        self.dropout = self.lfc.create_layer('dropout')
        
        # layers
        self._layers = []
        self.layers_params = []
        self.a = []
        self.h = []
        # input
        self.conv1 = self.lfc.create_layer('convIn_64')
        self._layers.append(self.conv1)
        self._layers_names.append('conv')
        # conv2
        self.conv2 = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('conv64_64'),
        )
        self._layers.append(self.conv2)
        self._layers_names.append('conv')
        # conv3
        self.conv3 = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('conv64_128'),
        )
        self._layers.append(self.conv3)
        self._layers_names.append('conv')
        # fully connected
        self.fc = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('flatten'),
            self.lfc.create_layer('fcWidth')
        )
        self._layers.append(self.fc)
        self.layers_params.append('fc')
        # out
        self.out = nn.Sequential(
            self.dropout,   
            self.lfc.create_layer('fc128_Out')
        )
        self._layers.append(self.out)
        self.layers_params.append('fc')


    def forward(self, x):
        self.a = []
        self.h = []
        input_ = x

        for l in range(len(self._layers)):
            self.h.append(input_)
            a_l = self._layers[l](input_)
            input_ = self.activation_fn(a_l)
            if a_l.requires_grad:
                a_l.retain_grad() 
            self.a.append(a_l)

        return self.a[len(self.a)-1]


class DepthWidthCNN(nn.Module):
    def __init__(self, input_shape=1, output_shape=10, activation_fn=nn.ReLU, p=0.5, dataset="mnist"):
        super(DepthWidthCNN, self).__init__()
        # factory
        self.lfc = LayersFactory(input_shape, output_shape, p, dataset)
        # unique
        self.activation_fn = activation_fn()
        self.pool = self.lfc.create_layer('pool')
        self.dropout = self.lfc.create_layer('dropout')
        
        # layers
        self._layers = []
        self.layers_params = []
        self.a = []
        self.h = []
        # input
        self.conv1 = self.lfc.create_layer('convIn_64')
        self._layers.append(self.conv1)
        self._layers_names.append('conv')
        # conv2
        self.conv2 = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('conv64_64'),
        )
        self._layers.append(self.conv2)
        self._layers_names.append('conv')
        # conv3
        self.conv3 = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('conv64_64'),
        )
        self._layers.append(self.conv3)
        self._layers_names.append('conv')
        # conv4
        self.conv4 = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('conv64_64'),
        )
        self._layers.append(self.conv4)
        self._layers_names.append('conv')
        # conv5
        self.conv5 = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('conv64_128'),
        )
        self._layers.append(self.conv5)
        self._layers_names.append('conv')
        # conv6
        self.conv6 = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('conv128_128'),
        )
        self._layers.append(self.conv6)
        self._layers_names.append('conv')
        # fully connected
        self.fc = nn.Sequential(
            self.dropout,
            self.pool,
            self.lfc.create_layer('flatten'),
            self.lfc.create_layer('fcDepthWidth')
        )
        self._layers.append(self.fc)
        self.layers_params.append('fc')
        # out
        self.out = nn.Sequential(
            self.dropout,
            self.lfc.create_layer('fc128_Out')
        )
        self._layers.append(self.out)
        self.layers_params.append('fc')


    def forward(self, x):
        self.a = []
        self.h = []
        input_ = x

        for l in range(len(self._layers)):
            self.h.append(input_)
            a_l = self._layers[l](input_)
            input_ = self.activation_fn(a_l)
            if a_l.requires_grad:
                a_l.retain_grad() 
            self.a.append(a_l)

        return self.a[len(self.a)-1]
