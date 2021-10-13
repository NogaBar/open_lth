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
    def branch_function(self, seed: int, property:str = 'lth', path:str =''):
        # Randomize the mask.
        orig_mask = Mask.load(self.level_root)
        start_step = self.lottery_desc.str_to_step('0ep')
        # Use level 0 model for dense pre-pruned model
        if not get_platform().is_primary_process: return
        base_model = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'), start_step,
                                          self.lottery_desc.model_hparams)
        orig_model = PrunedModel(base_model, Mask.ones_like(base_model))
        model_graduate = copy.deepcopy(orig_model)
        lth_model = PrunedModel(copy.deepcopy(base_model), orig_mask)

        # Randomize while keeping the same layerwise proportions as the original mask.
        prunable_tensors = set(orig_model.prunable_layer_names) - set(orig_model.prunable_conv_names)
        tensors = {k[6:]: v.clone() for k, v in orig_model.state_dict().items() if k[6:] in prunable_tensors}

        train_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=True)
        input = list(train_loader)[0][0]

        if property == 'lth':
            with torch.no_grad():
                lth_features = lth_model.intermediate(input)
                norm = lth_features[-1].norm(p=2)
                print(f'initial norm: {norm.item()}')

            for n, p in lth_model.model.named_parameters():
                # if 'bias' in n:
                #     continue
                p.data = p.data / torch.sqrt(norm)

            with torch.no_grad():
                final_norm = lth_model.intermediate(input)[-1].norm(p=2)
                print(f'final norm: {final_norm}')

            train.standard_train(lth_model, self.branch_root, self.lottery_desc.dataset_hparams,
                                 self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)
        else:
            file_path = os.path.join(self.level_root, '../', path, f'properties_{property}.log')
            prop_values = np.load(file_path, allow_pickle=True)

            seeds = [np.argmax(prop_values[i, :]) for i in range(prop_values.shape[0])]
            curr_mask = Mask()
            for i, (name, param) in enumerate(tensors.items()):
                curr_mask[name] = shuffle_tensor(orig_mask[name], int(seed + seeds[i])).int()
                model_graduate.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
            with torch.no_grad():
                features = model_graduate.intermediate(input)
                norm = features[-1].norm(p=2)
            for n, p in model_graduate.model.named_parameters():
                p.data = p.data / torch.sqrt(norm)
            with torch.no_grad():
                final_norm = model_graduate.intermediate(input)[-1].norm(p=2)
                print(f'initial norm: {norm}')
                print(f'final norm: {final_norm}')

            train.standard_train(model_graduate, self.branch_root, self.lottery_desc.dataset_hparams,
                                 self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)




    @staticmethod
    def description():
        return "Plot singular values."

    @staticmethod
    def name():
        return 'normalized_init'
