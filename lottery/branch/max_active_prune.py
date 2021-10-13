# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import generate_mask_active, erank
import datasets.registry
import numpy as np
import copy
from platforms.platform import get_platform
from foundations import paths



class Branch(base.Branch):
    def branch_function(self, seed: int, property: str, trials: int):
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

        # Randomize while keeping the same layerwise proportions as the original mask.
        prunable_tensors = set(orig_model.prunable_layer_names) - set(orig_model.prunable_conv_names)
        tensors = {k[6:]: v.clone() for k, v in orig_model.state_dict().items() if k[6:] in prunable_tensors}

        train_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=True)
        input = list(train_loader)[0][0]
        features = orig_model.intermediate(input)
        features.insert(0, input)
        properties = np.zeros((len(tensors), trials))

        for i, (name, param) in enumerate(tensors.items()):
            curr_mask = Mask()
            for t in range(trials):
                curr_mask[name] = generate_mask_active(param, orig_mask[name].float().mean().item(), seed + t,
                                                       features[i]).int()
                if property == 'weight_erank':
                    properties[i, t] = erank(param * curr_mask[name])
                elif property == 'features_erank':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    # model[name].data = param * curr_mask[name]
                    features_for_erank = model.intermediate(input)
                    properties[i, t] = erank(features_for_erank[i])
            best_mask[name] = generate_mask_active(param, orig_mask[name].float().mean().item(), seed + np.argmax(properties[i, :]),
                                                       features[i]).int()
            model_graduate.register_buffer(PrunedModel.to_mask_name(name), best_mask[name].float())

        # Save the properties.
        best_mask.save(self.branch_root)
        if not get_platform().exists(self.branch_root): get_platform().makedirs(self.branch_root)

        with open(paths.properties(self.branch_root, property), 'wb') as f:
            np.save(f, properties)

        # Train the model with the new mask.
        pruned_model = PrunedModel(base_model, best_mask)
        train.standard_train(pruned_model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly active prune the model with max property."

    @staticmethod
    def name():
        return 'max_active_prune'
