# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """A residual neural network as originally designed for CIFAR-10."""

    class ConvModule(nn.Module):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters):
            super(Model.ConvModule, self).__init__()
            self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            # self.bn = nn.BatchNorm2d(out_filters)

        def forward(self, x):
            return F.relu(self.conv(x))

    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__()
        outputs = outputs or 100

        layers = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Model.ConvModule(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layers)
        input_fc = 1
        if len(plan) == 3:
            input_fc = 16384
        elif len(plan) == 6:
            input_fc = 8192
        elif len(plan) == 9:
            input_fc = 4096



        fc_layers = [nn.Linear(input_fc, 256), nn.Linear(256, 256)]
        self.fc_layers = nn.ModuleList(fc_layers)
        self.fc = nn.Linear(256, outputs)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = F.relu(fc_layer(x))
        x = self.fc(x)
        return x

    def intermediate(self, x, conv_layers=False, no_activation=False):
        features = []

        for l in self.layers:
            x = l(x)
            if conv_layers and isinstance(l, Model.ConvModule):
                features.append(x.view(x.size(0), -1))


        features.append(x.view(x.size(0), -1))
        for fc_layer in self.fc_layers:
            features.append(F.relu(fc_layer(features[-1])))
        features.append(self.fc(features[-1]))
        return features

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar100_conv') and
                model_name[-1] in {'2', '4', '6'})

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=100):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        num = int(model_name[-1])
        if num == 2:
            plan = [64, 64, 'M']
        elif num == 4:
            plan = [64, 64, 'M', 128, 128, 'M']
        elif num == 6:
            plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M']
        else:
            raise ValueError('Unknown conv model: {}'.format(model_name))

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar100_conv6',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar100',
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            lr=0.1,
            training_steps='40ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_conv=0.15,
            pruning_fraction_last_fc=0.1
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
