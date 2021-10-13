# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform


class Dataset(base.ImageDataset):
    """Random dataset."""

    @staticmethod
    def num_train_examples(): return 60000

    @staticmethod
    def num_test_examples(): return 10000

    @staticmethod
    def num_classes(): return 10

    @staticmethod
    def get_train_set(use_augmentation):
        # No augmentation for random data.
        train_set = torchvision.datasets.FakeData(size=60000, image_size=(28, 28), num_classes=10, random_offset=259,
                                                  transform=torchvision.transforms.ToTensor())
        # mnist_train_set = torchvision.datasets.MNIST(
        #     train=True, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
        return None, None, train_set

    @staticmethod
    def get_test_set():
        test_set = torchvision.datasets.FakeData(size=10000, image_size=(28, 28), num_classes=10,
                                                 random_offset=259 + 60000, transform=torchvision.transforms.ToTensor())
        return None, None, test_set

    def __init__(self,  examples, labels, dataset=None):
        tensor_transforms = torchvision.transforms.Compose[torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(mean=[0.286], std=[0.3205])]
        super(Dataset, self).__init__(examples, labels, dataset, [], tensor_transforms)

    def example_to_image(self, example):
        return Image.fromarray(example.numpy(), mode='L')


DataLoader = base.DataLoader
