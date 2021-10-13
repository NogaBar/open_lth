# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import vectorize, unvectorize, shuffle_tensor, shuffle_state_dict, weight_erank, feature_erank, \
    activation, condition, mutual_coherence, min_singular, activation_mean, gradient_mean

from platforms.platform import get_platform
from foundations import paths
import json
import os
import datasets.registry


class Branch(base.Branch):
    def branch_function(self, seed: int, property: str = 'weight_erank',
                        layers_to_ignore: str = '', conv_layers: bool = False,
                        cross_lt_path: str = '', cross_data: str = '', cross_fake_path: str = '',
                        add_rand_data: bool = False):
        # Randomize the mask.
        mask = Mask.load(self.level_root)
        start_step = self.lottery_desc.str_to_step('0ep')
        model = PrunedModel(models.registry.load(self.level_root, start_step, self.lottery_desc.model_hparams), mask)
        prunable_tensors = set(model.prunable_layer_names)
        tensors = {k: v for k, v in model.state_dict().items() if k[6:] in prunable_tensors}
        properties = {}

        for dir in os.listdir(os.path.join(self.level_root, '../')):
            if 'randomly_prune' in dir:
                rand_dir = dir
        rand_path = os.path.join(self.level_root, '../', rand_dir)

        rand_mask = Mask.load(rand_path)
        rand_model = PrunedModel(models.registry.load(rand_path, start_step, self.lottery_desc.model_hparams), rand_mask)

        # Calculate effective rank of LTH weights
        if property == 'weight_erank':
            properties['lth'] = weight_erank(tensors)
            properties['random'] = weight_erank({k: v for k, v in rand_model.state_dict().items() if k[6:] in prunable_tensors})
        elif property == 'weight_frobenius':
            properties['lth'] = {k: torch.norm(v).item() for k, v in tensors.items()}
            properties['random'] = {k: torch.norm(v).item() for k, v in rand_model.state_dict().items() if k[6:] in prunable_tensors}
        elif property == 'weight_condition':
            properties['lth'] = {k: condition(v) for k, v in tensors.items()}
            properties['random'] = {k: condition(v) for k, v in rand_model.state_dict().items() if k[6:] in prunable_tensors}
        elif property == 'weight_mutual_coherence':
            properties['lth'] = {k: mutual_coherence(v) for k, v in tensors.items()}
            properties['random'] = {k: mutual_coherence(v) for k, v in rand_model.state_dict().items() if k[6:] in prunable_tensors}
        elif property == 'weight_min_singular':
            properties['lth'] = {k: min_singular(v) for k, v in tensors.items()}
            properties['random'] = {k: min_singular(v) for k, v in rand_model.state_dict().items() if k[6:] in prunable_tensors}
        elif property == 'features_erank':
            train_loader = datasets.registry.get( self.lottery_desc.dataset_hparams, train=True)
            input = list(train_loader)[0][0]
            properties['lth'] = feature_erank(model, input, conv_layers)
            properties['random'] = feature_erank(rand_model, input, conv_layers)
        elif property == 'gradients_mean':
            train_loader = datasets.registry.get( self.lottery_desc.dataset_hparams, train=True)
            input, labels = list(train_loader)[0]
            properties['lth'] = gradient_mean(model, input, labels, conv_layers)
            properties['random'] = gradient_mean(rand_model, input, labels, conv_layers)

            if add_rand_data:
                rand_input = torch.rand_like(input) * input.var() + input.mean()
                rand_input = rand_input * input.norm() / rand_input.norm()
                properties['lth_rand_data'] = feature_erank(model, rand_input, conv_layers)
                properties['random_rand_data'] = feature_erank(rand_model, rand_input, conv_layers)

        elif property == 'features_frobenius':
            train_loader = datasets.registry.get( self.lottery_desc.dataset_hparams, train=True)
            input = list(train_loader)[0][0]
            properties['lth'] = [f.norm().item() for f in model.intermediate(input, conv_layers)]
            properties['random'] = [f.norm().item() for f in rand_model.intermediate(input, conv_layers)]

            if add_rand_data:
                rand_input = torch.rand_like(input) * input.var() + input.mean()
                rand_input = rand_input * input.norm() / rand_input.norm()
                properties['lth_rand_data'] = [f.norm().item() for f in model.intermediate(rand_input, conv_layers)]
                properties['random_rand_data'] = [f.norm().item() for f in rand_model.intermediate(rand_input, conv_layers)]

        elif property == 'features_mutual_coherence':
            train_loader = datasets.registry.get( self.lottery_desc.dataset_hparams, train=True)
            input = list(train_loader)[0][0]
            properties['lth'] = [mutual_coherence(f) for f in model.intermediate(input, conv_layers)]
            properties['random'] = [mutual_coherence(f) for f in rand_model.intermediate(input, conv_layers)]
        elif property == 'features_min_singular':
            train_loader = datasets.registry.get( self.lottery_desc.dataset_hparams, train=True)
            input = list(train_loader)[0][0]
            properties['lth'] = [min_singular(f) for f in model.intermediate(input, conv_layers)]
            properties['random'] = [min_singular(f) for f in rand_model.intermediate(input, conv_layers)]
        elif property == 'activation':
            train_loader = datasets.registry.get( self.lottery_desc.dataset_hparams, train=True)
            input = list(train_loader)[0][0]
            properties['lth'] = activation_mean(model, input, conv_layers)
            properties['random'] = activation_mean(rand_model, input, conv_layers)
        # Error.
        else: raise ValueError(f'Invalid property: {property}')

        if cross_lt_path != '' and cross_fake_path != '':
            path = os.path.join(cross_lt_path, f'level_{self.level}', 'main')
            cross_mask = Mask.load(path)
            cross_model = PrunedModel(models.registry.load(path, self.lottery_desc.str_to_step('0ep'),
                                                           self.lottery_desc.model_hparams), cross_mask)

            path = os.path.join(cross_fake_path, f'level_{self.level}', 'main')
            fake_mask = Mask.load(path)
            fake_model = PrunedModel(models.registry.load(path, self.lottery_desc.str_to_step('0ep'),
                                                           self.lottery_desc.model_hparams), fake_mask)

            if property == 'features_erank':
                properties['cross_lth'] = feature_erank(cross_model, input, conv_layers)
                properties['fake_lth'] = feature_erank(fake_model, input, conv_layers)
                if add_rand_data:
                    rand_input = torch.rand_like(input) * input.var() + input.mean()
                    rand_input = rand_input * input.norm() / rand_input.norm()
                    properties['cross_lth_rand_data'] = feature_erank(cross_model, rand_input, conv_layers)
                    properties['fake_lth_rand_data'] = feature_erank(fake_model, rand_input, conv_layers)

            elif property == 'features_frobenius':
                properties['cross_lth'] = [f.norm().item() for f in cross_model.intermediate(input, conv_layers)]
                properties['fake_lth'] = [f.norm().item() for f in fake_model.intermediate(input, conv_layers)]
                if add_rand_data:
                    rand_input = torch.rand_like(input) * input.var() + input.mean()
                    rand_input = rand_input * input.norm() / rand_input.norm()
                    properties['cross_lth_rand_data'] = [f.norm().item()
                                                         for f in cross_model.intermediate(rand_input, conv_layers)]
                    properties['fake_lth_rand_data'] = [f.norm().item()
                                                        for f in fake_model.intermediate(rand_input, conv_layers)]


        # Save model
        if not get_platform().is_primary_process: return
        if not get_platform().exists(self.branch_root): get_platform().makedirs(self.branch_root)

        with open(paths.properties(self.branch_root, property), 'w') as f:
            json.dump(properties, f)





    @staticmethod
    def description():
        return "Calculate properties of lottery ticket."

    @staticmethod
    def name():
        return 'properties'
