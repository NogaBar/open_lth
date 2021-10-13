# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import generate_mask_active, shuffle_state_dict
import datasets.registry
from platforms.platform import get_platform
import copy

class Branch(base.Branch):
    def branch_function(self, seed: int, active_conv: bool = False):
        # Randomize the mask.
        mask = Mask.load(self.level_root)
        start_step = self.lottery_desc.str_to_step('0ep')
        # Use level 0 model for dense pre-pruned model
        if not get_platform().is_primary_process: return
        base_model = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'), start_step, self.lottery_desc.model_hparams)
        model = PrunedModel(base_model, Mask.ones_like(base_model))

        # Randomize while keeping the same layerwise proportions as the original mask.
        prunable_tensors = set(model.prunable_layer_names) - set(name[6:]  for name in model.prunable_conv_names if 'model' in name)
        tensors = {k[6:]: v for k, v in model.state_dict().items() if k[6:] in prunable_tensors}

        if not active_conv:
            mask_conv = copy.deepcopy(mask)
            for k, p in mask.items():
                if k not in set([k[6:] for k in model.prunable_conv_names]):
                    mask_conv.pop(k)
            new_mask_conv = Mask(shuffle_state_dict(mask_conv, seed=seed))
        else:
            raise NotImplementedError

        train_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=True)
        input = list(train_loader)[0][0]
        with torch.no_grad():
            features = model.intermediate(input)
            if len(model.prunable_conv_names) == 0:
                features.insert(0, input)

            for i, (name, param) in enumerate(tensors.items()):
                mask[name] = generate_mask_active(param, mask[name].float().mean().item(), seed, features[i]).int()

        for n, m in new_mask_conv.items():
            mask[n] = m

        # Save the new mask.
        mask.save(self.branch_root)

        # Train the model with the new mask.
        if not get_platform().is_primary_process: return
        base_model = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'), start_step,
                                          self.lottery_desc.model_hparams)
        pruned_model = PrunedModel(base_model, mask)
        train.standard_train(pruned_model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly active prune the model."

    @staticmethod
    def name():
        return 'active_random_prune'
