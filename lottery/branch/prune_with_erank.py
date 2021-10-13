# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import erank, shuffle_state_dict, mutual_coherence, shuffle_tensor
import datasets.registry
from platforms.platform import get_platform
import copy
import torch.optim as optim
from tqdm import tqdm
import os
import json

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pruning.sparse_global

def erank_tensor(M):
    U, S, Vt = torch.svd(M, compute_uv=True, some=True)
    new_S = torch.zeros_like(S) + S
    new_S[S <= 1e-6] = 0
    normalized = new_S / new_S.sum()
    return -(normalized[normalized > 0] * torch.log(normalized[normalized > 0])).sum()

def mutual_coherence_tensor(M):
    W = torch.nn.functional.normalize(M, dim=1)
    I = torch.eye(W.size()[0], device=W.device)
    W_adj = W @ W.T - I
    return W_adj.abs().max()

def train_property(train_loader, steps, model, optimizer, algorithm, features_norm_coeff, weight_reg_coeff,
                   log_erank_freq, test_input):
    norms = []
    eranks = []
    for step, (input, _) in tqdm(enumerate(train_loader)):
        if step >= steps:
            break
        optimizer.zero_grad()
        input = input.to(device=get_platform().torch_device)
        output = model(input)

        if algorithm == 'erank':
            l = -erank_tensor(output)
        elif algorithm == 'erank_norm':
            if output.norm() > 5000:
                l = torch.tensor(0., requires_grad=True)
            else:
                l = -erank_tensor(output) - features_norm_coeff * output.norm()
        elif algorithm == 'features_frobenius':
            l = -output.norm()
        elif algorithm == 'features_nuclear':
            l = -output.norm(p=1)
        elif algorithm == 'mutual_coherence':
            l = mutual_coherence_tensor(output)
        elif algorithm == 'srank':  # frobenious / spectral
            l = -(output.norm() / output.norm(p=2))
        else:
            raise NotImplementedError
        if weight_reg_coeff < 0:
            for n, p in model.named_parameters():
                l += torch.norm(p) * weight_reg_coeff
        l.backward()
        optimizer.step()

        if step % log_erank_freq == 0:
            with torch.no_grad():
                output_test = model(test_input)
                norms.append(output_test.norm().item())
                eranks.append(erank(output_test))
                print(f'step {step}: norm: {norms[-1]} erank: {eranks[-1]}')
    return norms, eranks



class Branch(base.Branch):
    def branch_function(self, steps: int = 100, init_lr: float = 0.1,
                        weight_reg_coeff: float = 0., num_batches: int = 1,
                        algorithm: str = 'erank', features_norm_coeff: float = 0., init_momentum: float = 0.,
                        log_erank_freq: int = 100, one_shot_erank: bool = False):

        start_step = self.lottery_desc.str_to_step('0ep')
        final_step = self.lottery_desc.str_to_step('40ep')
        train_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=True)
        test_input = list(train_loader)[0][0].to(device=get_platform().torch_device)

        if not one_shot_erank:
            # Upload previus mask and model and Prune
            if self.level > 0:
                prev_mask = Mask.load(self.branch_root.replace(f'level_{self.level}', f'level_{self.level-1}'))
                prev_model_erank = models.registry.load(self.branch_root.replace(f'level_{self.level}', f'level_{self.level-1}'), final_step,
                                                        self.lottery_desc.model_hparams)
                new_mask = pruning.sparse_global.Strategy.prune(self.lottery_desc.pruning_hparams, prev_model_erank, prev_mask)
            else:
                # use initial model and ones mask
                prev_model_erank = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'), start_step,
                                                        self.lottery_desc.model_hparams)
                new_mask = Mask.ones_like(prev_model_erank)
        else:
            prev_mask = Mask.load(self.level_root)
            new_hparam = copy.deepcopy(self.lottery_desc.pruning_hparams)
            new_hparam.pruning_fraction_last_fc = \
                1. - ((prev_mask['fc.weight'] > 0).sum() / prev_mask['fc.weight'].numel()).item()
            total_fc = 0
            remaining_fc = 0
            total_conv = 0
            remaining_conv = 0
            for k in prev_mask.keys():
                if 'fc' in k and k != 'fc.weight':
                    total_fc += prev_mask[k].numel()
                    remaining_fc += prev_mask[k].sum()
                elif 'conv' in k:
                    total_conv += prev_mask[k].numel()
                    remaining_conv += prev_mask[k].sum()
            new_hparam.pruning_fraction = 1. - (remaining_fc / total_fc).item()
            if total_conv > 0:
                new_hparam.pruning_conv = 1. - (remaining_conv / total_conv).item()
            prev_model_erank = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'),
                                                    start_step,
                                                    self.lottery_desc.model_hparams)
            init_prune_model = PrunedModel(prev_model_erank, Mask.ones_like(prev_model_erank)).to(device=get_platform().torch_device)
            # train with erank
            optimizer = optim.SGD(init_prune_model.parameters(), init_lr,
                                  weight_decay=weight_reg_coeff ,
                                  momentum=init_momentum)
            init_erank = erank(init_prune_model(test_input))
            init_norm = init_prune_model(test_input).norm().item()
            norms, eranks = train_property(train_loader, steps, init_prune_model, optimizer, algorithm,
                                           features_norm_coeff, weight_reg_coeff, log_erank_freq, test_input)

            new_mask = pruning.sparse_global.Strategy.prune(new_hparam, prev_model_erank,
                                                            Mask.ones_like(prev_model_erank))



        new_mask.save(self.branch_root)
        base_model = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'), start_step,
                                          self.lottery_desc.model_hparams)
        if not one_shot_erank:
            base_model_cp = copy.deepcopy(base_model)
            # model that will be trained with ernak
            init_prune_model = PrunedModel(base_model_cp, new_mask)


            init_prune_model = init_prune_model.to(device=get_platform().torch_device)
            if weight_reg_coeff >= 0:
                optimizer = optim.SGD(init_prune_model.parameters(), init_lr,
                                      weight_decay=weight_reg_coeff * (new_mask.density ** (1./len(init_prune_model.prunable_layer_names))),
                                      momentum=init_momentum)
                # optimizer = optim.SGD(init_prune_model.parameters(), init_lr, weight_decay=weight_reg_coeff, momentum=init_momentum)
            else:
                optimizer = optim.SGD(init_prune_model.parameters(), init_lr, weight_decay=0., momentum=init_momentum)



            with torch.no_grad():
                init_norm = init_prune_model.intermediate(test_input, no_activation=True)[-1].norm().item()
                init_erank = erank(init_prune_model.intermediate(test_input, no_activation=True)[-1])

                print('initial norm', init_norm)
                print('initial erank', init_erank)

            norms, eranks = train_property(train_loader, steps, init_prune_model, optimizer, algorithm, features_norm_coeff,
                                           weight_reg_coeff, log_erank_freq, test_input)





        if not get_platform().is_primary_process: return
        if not get_platform().exists(self.branch_root): get_platform().makedirs(self.branch_root)
        with open(os.path.join(self.branch_root, 'properties.json'), 'w') as f:
            json.dump({'init_norm': init_norm, 'init_erank': init_erank,
                       'training norms': norms, 'training eranks': eranks
                       }, f)

        train.standard_train(PrunedModel(base_model, new_mask), self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)
        # overwrite the the model with CE with the one trained with erank
        init_prune_model.save(self.branch_root, final_step)


    @staticmethod
    def description():
        return "Train initial weights with random masks"

    @staticmethod
    def name():
        return 'prune_with_erank'
