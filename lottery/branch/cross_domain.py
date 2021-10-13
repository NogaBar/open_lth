# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from platforms.platform import get_platform
import os
import copy


class Branch(base.Branch):
    def branch_function(self, lth_path: str, lth_data: str):
        # load mask and model from different directory.
        path = os.path.join(lth_path, f'level_{self.level}', 'main')
        mask = Mask.load(path)

        start_step = self.lottery_desc.str_to_step('0ep')
        state_step = start_step

        if lth_data == 'cifar100':
            tmp_hparam = copy.deepcopy(self.lottery_desc.model_hparams)
            tmp_hparam.model_name = self.lottery_desc.model_hparams.model_name.replace('cifar', 'cifar100')
            model_in = models.registry.load(path, state_step, tmp_hparam)

            # change fc to match 10 classes
            model_in.fc = torch.nn.Linear(model_in.fc.in_features, 10)
            # change fc mask
            mask['fc.weight'] = torch.rand_like(model_in.fc.weight) < mask['fc.weight'].float().mean()
        # elif lth_data == 'cifar10':
        elif lth_data == 'cifar10' or lth_data == 'random_color':
            if not get_platform().is_primary_process: return
            tmp_hparam = copy.deepcopy(self.lottery_desc.model_hparams)
            tmp_hparam.model_name = self.lottery_desc.model_hparams.model_name.replace('cifar100', 'cifar')
            model_in = models.registry.load(path, state_step, tmp_hparam)

            # change fc to match 100 classes
            model_in.fc = torch.nn.Linear(model_in.fc.in_features, 100)
            # change fc mask
            mask['fc.weight'] = torch.rand_like(model_in.fc.weight) < mask['fc.weight'].float().mean()
        else:
            model_in = models.registry.load(path, state_step, self.lottery_desc.model_hparams)


        # Train the model with the new mask.
        model = PrunedModel(model_in, mask)
        train.standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Use lth of another dataset"

    @staticmethod
    def name():
        return 'cross_domain'
