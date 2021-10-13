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
import numpy as np
from utils.tensor_utils import generate_mask_active, erank, shuffle_tensor, mutual_coherence, activation_mean

class Branch(base.Branch):
    def branch_function(self, seed: int, erank_path: str = '', coherence_path: str = '',
                        frobenius_path: str = '', min_singular_path: str = '', nuclear_path: str = '',
                        stable_rank_path:str = '', normalized: bool = False, batch_average: int = 1):
        # Randomize the mask.
        orig_mask = Mask.load(self.level_root)
        best_mask = Mask()
        start_step = self.lottery_desc.str_to_step('0ep')
        # Use level 0 model for dense pre-pruned model
        if not get_platform().is_primary_process: return
        base_model = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'), start_step,
                                          self.lottery_desc.model_hparams)
        orig_model = PrunedModel(base_model, Mask.ones_like(base_model))
        model_graduate = copy.deepcopy(orig_model)
        model = copy.deepcopy(orig_model)
        lth_model = PrunedModel(copy.deepcopy(base_model), orig_mask)

        # Randomize while keeping the same layerwise proportions as the original mask.
        prunable_tensors = set(orig_model.prunable_layer_names) - set(orig_model.prunable_conv_names)
        tensors = {k[6:]: v.clone() for k, v in orig_model.state_dict().items() if k[6:] in prunable_tensors}

        train_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=True)
        input = []
        for b in range(batch_average):
            input.append(list(train_loader)[b+1][0])
        
        data = pd.DataFrame()

        # lth_features = lth_model.intermediate(input)
        # _, s, _ = torch.svd(lth_features[-1], compute_uv=False)
        # if normalized:
        #     s = s / s[0]
        # singular_values.append(s)

        eranks = np.load(os.path.join(self.level_root, '../', erank_path), allow_pickle=True)
        coherence = np.load(os.path.join(self.level_root, '../', coherence_path), allow_pickle=True)
        frobenius = np.load(os.path.join(self.level_root, '../', frobenius_path), allow_pickle=True)
        min_singular = np.load(os.path.join(self.level_root, '../', min_singular_path), allow_pickle=True)
        nuclear = np.load(os.path.join(self.level_root, '../', nuclear_path), allow_pickle=True)
        stable_ranks = np.load(os.path.join(self.level_root, '../', stable_rank_path), allow_pickle=True)
        erank_seeds = []
        coherence_seeds = []
        frobenius_seeds = []
        min_singular_seeds = []
        nuclear_seeds = []
        stable_rank_seeds = []

        for layer in range(eranks.shape[0]):
            erank_seeds.append(np.argmax(eranks[layer, :]))
            coherence_seeds.append(np.argmax(coherence[layer, :]))
            frobenius_seeds.append(np.argmax(frobenius[layer, :]))
            min_singular_seeds.append(np.argmax(min_singular[layer, :]))
            nuclear_seeds.append(np.argmax(nuclear[layer, :]))
            stable_rank_seeds.append(np.argmax(stable_ranks[layer, :]))

        types = ['lth', 'erank', 'mutual coherence', 'frobenius', 'min singular', 'nuclear', 'stable rank', 'random']

        # Assign all masks to model
        for b in range(batch_average):
            lth_features = lth_model.intermediate(input[b])
            if normalized:
                norm = torch.pow(lth_features[-1].norm(p=2), 1/3)
                lth_model.normalize_all_parameters(1./norm)
                lth_features = lth_model.intermediate(input[b])
                lth_model.normalize_all_parameters(norm)
            _, s, _ = torch.svd(lth_features[-1], compute_uv=False)
            # if normalized:
            #     s = s / s[0]
            data = data.append(pd.DataFrame({'type': ['lth'], 'property': 'erank', 'value': erank(lth_features[-1])}), ignore_index=True)
            data = data.append(pd.DataFrame({'type': ['lth'], 'property': 'mutual coherence',
                                      'value': mutual_coherence(lth_features[-1])}), ignore_index=True)
            data = data.append(pd.DataFrame({'type': ['lth'], 'property': 'frobenius',
                                      'value': torch.norm(lth_features[-1]).item()}), ignore_index=True)
            data = data.append(pd.DataFrame({'type': ['lth'], 'property': 'min svd', 'value': s[-1].item()}),
                               ignore_index=True)
            data = data.append(pd.DataFrame({'type': ['lth'], 'property': 'nuclear',
                                             'value': lth_features[-1].norm(p=1).item()}), ignore_index=True)
            data = data.append(pd.DataFrame({'type': ['lth'], 'property': 'activation',
                                             'value': activation_mean(lth_model, input[b])}), ignore_index=True)
            data = data.append(pd.DataFrame({'type': ['lth'], 'property': 'stable rank',
                                             'value':
                                            (torch.linalg.norm(lth_features[-1], ord='fro') / torch.linalg.norm(lth_features[-1],ord=2)).item()}), ignore_index=True)
            for seeds, type in zip([erank_seeds, coherence_seeds, frobenius_seeds, min_singular_seeds, nuclear_seeds, stable_rank_seeds, [seed] * len(erank_seeds)], types[1:]):
                curr_mask = Mask()
                for i, (name, param) in enumerate(tensors.items()):
                    curr_mask[name] = shuffle_tensor(orig_mask[name], int(seed + seeds[i])).int()
                    model_graduate.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                features = model_graduate.intermediate(input[b])
                if normalized:
                    model_graduate.normalize_all_parameters(1. / torch.pow(features[-1].norm(p=2), 1/3))
                    features = model_graduate.intermediate(input[b])
                _, s, _ = torch.svd(features[-1], compute_uv=False)
                # if normalized:
                #     s = s / s[0]
                data = data.append(pd.DataFrame({'type': [type], 'property': 'erank', 'value': erank(features[-1])}), ignore_index=True)
                data = data.append(pd.DataFrame({'type': [type], 'property': 'mutual coherence',
                                          'value': mutual_coherence(features[-1])}), ignore_index=True)
                data = data.append(pd.DataFrame({'type': [type], 'property': 'frobenius',
                                          'value': torch.norm(features[-1]).item()}), ignore_index=True)
                data = data.append(pd.DataFrame({'type': [type], 'property': 'min svd', 'value': s[-1].item()}), ignore_index=True)
                data = data.append(pd.DataFrame({'type': [type], 'property': 'nuclear',
                                                 'value': features[-1].norm(p=1).item()}), ignore_index=True)
                data = data.append(pd.DataFrame({'type': [type], 'property': 'activation',
                                                 'value': activation_mean(model_graduate, input[b])}), ignore_index=True)
                data = data.append(pd.DataFrame({'type': [type], 'property': 'stable rank',
                                                 'value':
                                                (torch.linalg.norm(features[-1], ord='fro') / torch.linalg.norm(features[-1],ord=2)).item()}), ignore_index=True)
                model_graduate = copy.deepcopy(orig_model)

        data.to_csv(os.path.join(self.branch_root, 'cross_properties.csv'))



    @staticmethod
    def description():
        return "Save cross properties."

    @staticmethod
    def name():
        return 'cross_properties'
