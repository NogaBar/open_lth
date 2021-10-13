# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import shuffle_state_dict, weight_erank, feature_erank, activation, generate_mask_active, features_spectral, features_frobenius, features_spectral_fro_ratio, erank

from platforms.platform import get_platform
from foundations import paths
import json
import os
import datasets.registry
import copy

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import pandas as pd

class Branch(base.Branch):
    def branch_function(self, seed: int, property: str = 'features_erank',
                        trials: int = 10000, conv_layers: bool = False, rand_data: str = 'natural',
                        no_activation: bool= False,
                        cross_domain_path: str = 'none',
                        cross_domain_data: str = 'none',
                        path_all_trials: str = 'none'):
        # Randomize the mask.
        mask = Mask.load(self.level_root)
        start_step = self.lottery_desc.str_to_step('0ep')
        base_model = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'), start_step,
                                          self.lottery_desc.model_hparams)
        orig_model = PrunedModel(copy.deepcopy(base_model), Mask.ones_like(base_model))
        lth_model = PrunedModel(copy.deepcopy(base_model), mask)
        prunable_tensors = set(orig_model.prunable_layer_names)
        orig_tensors = {k: v for k, v in orig_model.state_dict().items() if k[6:] in prunable_tensors and
                        k not in orig_model.prunable_conv_names}
        lth_tensors = {k: v for k, v in lth_model.state_dict().items() if k[6:] in prunable_tensors}
        rand_properties = []
        active_properties = []

        if path_all_trials == 'none':
            if not get_platform().exists(paths.properties(self.branch_root, property)):
                train_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=True)
                input = list(train_loader)[0][0]
                if rand_data == 'uniform':
                    input = 2 * torch.rand_like(input) - 1
                elif rand_data == 'gaussian':
                    input = torch.randn_like(input)

                # Calculate effective rank of LTH
                if property == 'weight_erank':
                    lth_properties = [erank(v) for k, v in lth_tensors.items()]
                elif property == 'weight_frobenius':
                    lth_properties = [v.norm().item() for k, v in lth_tensors.items()]
                elif property == 'features_erank':
                    lth_properties = feature_erank(lth_model, input, conv_layers, no_activation)
                elif property == 'activation':
                    train_loader = datasets.registry.get( self.lottery_desc.dataset_hparams, train=True)
                    input = list(train_loader)[0][0]
                    lth_properties = activation(lth_model, input, conv_layers, no_activation)
                elif property == 'features_spectral':
                    lth_properties = features_spectral(lth_model, input, conv_layers, no_activation)
                elif property == 'features_frobenius':
                    lth_properties = features_frobenius(lth_model, input, conv_layers, no_activation)
                elif property == 'features_spectral_fro_ratio':
                    lth_properties = features_spectral_fro_ratio(lth_model, input, conv_layers, no_activation)
                # Error.
                else: raise ValueError(f'Invalid property: {property}')

                cross_domain_prop = None
                if cross_domain_path != 'none':
                    # load model + mask
                    path = os.path.join(cross_domain_path, f'level_{self.level}', 'main')
                    cross_mask = Mask.load(path)

                    start_step = self.lottery_desc.str_to_step('0ep')
                    state_step = start_step
                    if cross_domain_data == 'cifar100':
                        self.lottery_desc.model_hparams.model_name = self.lottery_desc.model_hparams.model_name.replace('cifar', 'cifar100')
                    elif cross_domain_data == 'cifar10':
                        self.lottery_desc.model_hparams.model_name = self.lottery_desc.model_hparams.model_name.replace('cifar100',
                                                                                                                        'cifar')
                    cross_model = PrunedModel(models.registry.load(path, state_step, self.lottery_desc.model_hparams), cross_mask)

                    if property == 'features_erank':
                        cross_domain_prop = feature_erank(cross_model, input, conv_layers, no_activation)
                    elif property == 'activation':
                        cross_domain_prop = activation(cross_model, input, conv_layers, no_activation)
                    elif property == 'features_spectral':
                        cross_domain_prop = features_spectral(cross_model, input, conv_layers, no_activation)
                    elif property == 'features_frobenius':
                        cross_domain_prop = features_frobenius(cross_model, input, conv_layers, no_activation)
                    elif property == 'features_spectral_fro_ratio':
                        cross_domain_prop = features_spectral_fro_ratio(cross_model, input, conv_layers, no_activation)

                # generate random masks
                for t in tqdm.tqdm(range(trials)):
                    # random mask
                    rand_mask = Mask(shuffle_state_dict(mask, seed=seed + t))
                    rand_model = PrunedModel(copy.deepcopy(base_model), rand_mask)

                    # curr_base_model = copy.deepcopy(base_model)
                    # active_mask = Mask.ones_like(curr_base_model)
                    # curr_model = PrunedModel(curr_base_model, active_mask)

                    # for i, (name, param) in enumerate(orig_tensors.items()):
                    #     name = name[6:]
                    #     features = [input] if len(orig_model.prunable_conv_names) == 0 else []
                    #     features.extend(curr_model.intermediate(input))
                    #     active_mask[name] = generate_mask_active(param, mask[name].float().mean().item(), seed+t, features[i]).int()
                    #     curr_model = PrunedModel(curr_base_model, active_mask)

                    if property == 'features_erank':
                        rand_properties.append(feature_erank(rand_model, input, conv_layers, no_activation))
                        # active_properties.append(feature_erank(curr_model, input, conv_layers))
                    elif property == 'activation':
                        rand_properties.append(activation(rand_model, input))
                        # active_properties.append(activation(curr_model, input))
                    elif property == 'weight_frobenius':
                        rand_properties.append([v.norm().item() for k, v in rand_model.state_dict().items() if k[6:] in prunable_tensors])
                    elif property == 'weight_erank':
                        rand_properties.append([erank(v) for k, v in rand_model.state_dict().items() if k[6:] in prunable_tensors])
                        # rand_properties.append(weight_erank({k: v for k, v in rand_model.state_dict().items() if k[6:] in prunable_tensors}))
                        # active_properties.append(weight_erank({k: v for k, v in curr_model.state_dict().items() if k[6:] in prunable_tensors}))
                    elif property == 'features_spectral':
                        rand_properties.append(features_spectral(rand_model, input, conv_layers, no_activation))
                    elif property == 'features_frobenius':
                        rand_properties.append(features_frobenius(rand_model, input, conv_layers, no_activation))
                    elif property == 'features_spectral_fro_ratio':
                        rand_properties.append(features_spectral_fro_ratio(rand_model, input, conv_layers, no_activation))
                # Save model
                if not get_platform().is_primary_process: return
                if not get_platform().exists(self.branch_root): get_platform().makedirs(self.branch_root)

                with open(paths.properties(self.branch_root, property), 'w') as f:
                    json.dump({'lth': lth_properties, 'random': rand_properties, 'active': rand_properties, 'cross_lth': cross_domain_prop}, f)

            else: # files already exits
                with open(paths.properties(self.branch_root, property), 'r') as f:
                    propeties_all = json.load(f)
                    lth_properties = propeties_all['lth']
                    rand_properties = propeties_all['random']
                    # rand_properties = [[p1, p2, p3, p4+0.05] for p1, p2, p3, p4 in rand_properties]
                    if 'cross_lth' in propeties_all.keys():
                        cross_domain_prop = propeties_all['cross_lth']
                    else:
                        cross_domain_prop = None

        else:
            # with open(path_all_trials) as f:
            #     log = json.load(f)
            # lth_properties = log['lth']
            import numpy as np
            log = np.load(path_all_trials)
            rand_properties = log.T
            if property == 'weight_erank':
                lth_properties = [erank(v) for k, v in lth_tensors.items()]
            cross_domain_prop = None
        new_labels = []
        layers = [i for i in range(len(rand_properties[0]))]
        if self.desc.lottery_desc.model_hparams.model_name == 'cifar_conv6':
            layers = [1, 4, 7, 10]
            new_labels = [
                    'conv1',
                    'conv4',
                    'conv6',
                    'last fc'
                ]
        if self.desc.lottery_desc.model_hparams.model_name == 'cifar_conv2':
            new_labels = [
                    'conv2',
                    'fc1',
                    'fc2',
                    'fc3'
                ]
        if self.desc.lottery_desc.model_hparams.model_name == 'cifar_vgg_19':
            layers = [3, 6, 11, 16, 21, 22]
            new_labels = [
                '1st pool',
                '2nd pool',
                '3rd pool',
                '4th pool',
                '5th pool',
                'fc',
            ]

        data = pd.concat(
            [pd.DataFrame({'layer': [layers[i]], property.replace('_', ' '): layer_prop})
             for prop in rand_properties for i, layer_prop in enumerate(prop)],
            ignore_index=True
        )
        sns.set_theme(style='white')
        # sns.set_palette("Reds")
        f = sns.displot(data=data, x=property.replace('_', ' '), hue='layer', bins=int(100), palette='dark', stat="probability")
        for i in range(len(rand_properties[0])):
            plt.axvline(lth_properties[i], linestyle='dashed', linewidth=1, color=f'C{i}')
            if cross_domain_prop:
                plt.axvline(cross_domain_prop[i], linewidth=1, color=f'C{i}')
        if len(new_labels) > 0:
            for t, x in zip(f.legend.texts, new_labels): t.set_text(x)
        f.fig.savefig(os.path.join(self.branch_root, f'property_{property}_sns.pdf'))




        # colors_list = ['blue', 'green', 'magenta', 'red', 'brown', 'cyan', 'purple', 'grey', 'orange', 'pink', 'lime']
        # for i in range(len(rand_properties[0])):
        #     rand_i = [p[i] for p in rand_properties]
        #     # active_i = [p[i] for p in active_properties]
        #     # plt.hist([rand_i, active_i], bins=int(trials/100), label=[f'random l{i+1}', f'active l{i+1}'])
        #     plt.hist(rand_i, bins=int(100), label=f'random l{i+1}', color=colors_list[i])
        #     plt.axvline(lth_properties[i], linestyle='dashed', linewidth=1, color=colors_list[i])
        #     if cross_domain_prop:
        #         plt.axvline(cross_domain_prop[i], linewidth=1, color=colors_list[i])

        # plt.legend()
        # plt.savefig(os.path.join(self.branch_root, f'property_{property}.pdf'))


    @staticmethod
    def description():
        return "Calculate histogram of properties."

    @staticmethod
    def name():
        return 'property_distribution'
