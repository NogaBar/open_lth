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
from utils.tensor_utils import generate_mask_active, erank, shuffle_tensor, mutual_coherence

class Branch(base.Branch):
    def branch_function(self, seed: int, erank_path: str = '', coherence_path: str = '',
                        frobenius_path: str = '', min_singular_path: str = ''):
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

        eranks = np.load(erank_path, allow_pickle=True)
        coherence = np.load(coherence_path, allow_pickle=True)
        frobenius = np.load(frobenius_path, allow_pickle=True)
        min_singular = np.load(min_singular_path, allow_pickle=True)
        erank_seeds = []
        coherence_seeds = []
        frobenius_seeds = []
        min_singular_seeds = []

        norms = [[]]
        # norms = []

        for name, param in lth_model.named_parameters():
            if name[6:] in prunable_tensors:
                norms[-1].append(param.norm().item() / param.numel())
                # norms[-1].append(param.norm().item())


        for layer in range(eranks.shape[0]):
            erank_seeds.append(np.argmax(eranks[layer, :]))
            coherence_seeds.append(np.argmax(coherence[layer, :]))
            frobenius_seeds.append(np.argmax(frobenius[layer, :]))
            min_singular_seeds.append(np.argmax(min_singular[layer, :]))

        # Assign all masks to model
        for seeds in [erank_seeds, coherence_seeds, frobenius_seeds, min_singular_seeds, [seed] * len(erank_seeds)]:
            curr_mask = Mask()
            norms.append([])
            for i, (name, param) in enumerate(tensors.items()):
                curr_mask[name] = shuffle_tensor(orig_mask[name], int(seed + seeds[i])).int()
                model_graduate.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
            model_graduate._apply_mask()
            for name, param in model_graduate.named_parameters():
                if name[6:] in prunable_tensors:
                    norms[-1].append(param.norm().item() / param.numel())
                    # norms[-1].append(param.norm().item())
            data_dist = pd.concat([
                pd.concat([pd.DataFrame(
                    {'weight': param.detach().numpy()[param.abs().detach().numpy() > 0], 'layer': name[6:]})]
                    , ignore_index=True)
                for name, param in model_graduate.named_parameters() if name[6:] in prunable_tensors], ignore_index=True)
            f = sns.histplot(data=data_dist, hue='layer', x='weight', stat="probability", common_norm=False)
            f.figure.savefig(os.path.join(self.branch_root, f'distribution_{seeds}.pdf'))
            f.figure.clf()
            model_graduate = copy.deepcopy(orig_model)

        types = [
            'lth',
            'erank', 'mutual coherence', 'frobenius', 'min singular', 'random']
        data = pd.concat([pd.DataFrame(
            {'norm': list(norms[i]), 'type': [types[i]] * len(norms[i]),
             'layer': list(range(len(norms[i])))}) for i in range(len(types))], ignore_index=True)
        #
        f = sns.lineplot(data=data, x='layer', y='norm', hue='type', markers=True, dashes=False, style="type")
        f.set(yscale='log')
        f.get_figure().savefig(os.path.join(self.branch_root, 'norms_layer.pdf'))
        f.get_figure().clf()

        data_dist = pd.concat([
            pd.concat([pd.DataFrame(
                {'weight': param.detach().numpy()[param.abs().detach().numpy() > 0], 'layer': name[6:]})]
             , ignore_index=True)
        for name, param in lth_model.named_parameters() if name[6:] in prunable_tensors], ignore_index=True)
        f = sns.histplot(data=data_dist, hue='layer', x='weight', stat="probability", common_norm=False)
        f.figure.savefig(os.path.join(self.branch_root, 'lth_distribution.pdf'))



    @staticmethod
    def description():
        return "Weight distribution along layers."

    @staticmethod
    def name():
        return 'weight_distribution'
