# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import erank, shuffle_state_dict, mutual_coherence, shuffle_tensor, generate_mask_active
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
from models.initializers import orthogonal

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


class Branch(base.Branch):
    def branch_function(self, seed: int, steps: int = 100, init_lr: float = 0.1,
                        weight_reg_coeff: float = 0., num_batches: int = 1,
                        algorithm: str = 'erank', features_norm_coeff: float = 0., init_momentum: float = 0.,
                        lth_dist: bool = False, path_max_random: str = '', path_rand_lt: str ='', normalized: bool = False,
                        kaiming_balance: bool = False, active_mask: bool = False, orthogonal_init: bool = False):
        # Randomize the mask.
        mask = Mask.load(self.level_root)
        mask = Mask(shuffle_state_dict(mask, seed=seed))
        start_step = self.lottery_desc.str_to_step('0ep')

        base_model = models.registry.load(self.level_root.replace(f'level_{self.level}', 'level_0'), start_step,
                                          self.lottery_desc.model_hparams)
        if orthogonal_init:
            base_model.apply(orthogonal)

        base_model_cp = copy.deepcopy(base_model)
        full_model = PrunedModel(base_model_cp, Mask.ones_like(base_model_cp))
        full_model = full_model.to(device=get_platform().torch_device)
        train_loader = datasets.registry.get(self.lottery_desc.dataset_hparams, train=True)
        input = list(train_loader)[0][0].to(device=get_platform().torch_device)

        if active_mask:
            with torch.no_grad():
                features = full_model.intermediate(input)
                if len(full_model.prunable_conv_names) == 0:
                    features.insert(0, input)

            for i, (name, param) in enumerate(full_model.named_parameters()):
                if name[6:] in full_model.prunable_layer_names:
                    mask[name[6:]] = generate_mask_active(param, mask[name[6:]].float().mean().item(), seed, features[i//2]).int().cpu()

        if lth_dist:
            orig_mask = Mask.load(self.level_root)
            for (n_s, p_s), (n, p) in zip(base_model.named_parameters(), full_model.named_parameters()):
                if n in base_model.prunable_layer_names:
                    p_s[mask[n]] = p[orig_mask[n]][torch.randperm(orig_mask[n].sum().int())]
        elif not path_max_random == '':
            rand_properties = np.load(os.path.join(self.level_root, '../', path_max_random), allow_pickle=True)
            seeds = []
            model = PrunedModel(base_model, Mask.ones_like(base_model))
            for layer in range(rand_properties.shape[0]):
                seeds.append(np.argmax(rand_properties[layer, :]))
            mask = Mask()
            orig_mask = Mask.load(self.level_root)
            i = 0
            for name, param in model.named_parameters():
                if name[6:] in model.prunable_layer_names:
                    mask[name[6:]] = shuffle_tensor(orig_mask[name[6:]], int(seed + seeds[i])).int()
                    model.register_buffer(PrunedModel.to_mask_name(name[6:]), mask[name[6:]].float())
                    i += 1
        elif not path_rand_lt == '':
            checkpoint = torch.load(os.path.join(self.level_root, '../', path_rand_lt, 'model_ep0_it0.pth'))
            rand_lt_model = copy.deepcopy(base_model)
            rand_lt_model.load_state_dict(checkpoint)
            mask = Mask()
            for n, p in rand_lt_model.named_parameters():
                if n in rand_lt_model.prunable_layer_names:
                    mask[n] = (p.abs() > 0).int()

        model = PrunedModel(base_model, mask)


        model = model.to(device=get_platform().torch_device)
        if weight_reg_coeff >= 0:
            optimizer = optim.SGD(model.parameters(), init_lr,
                                  weight_decay=weight_reg_coeff * (mask.density ** (1./len(model.prunable_layer_names))),
                                  momentum=init_momentum)
            # optimizer = optim.SGD(model.parameters(), init_lr, weight_decay=weight_reg_coeff, momentum=init_momentum)
        else:
            optimizer = optim.SGD(model.parameters(), init_lr, weight_decay=0., momentum=init_momentum)



        with torch.no_grad():
            init_norm = model.intermediate(input, no_activation=True)[-1].norm().item()
            init_erank = erank(model.intermediate(input, no_activation=True)[-1])
            init_mc = mutual_coherence(model.intermediate(input, no_activation=True)[-1])

            full_init_norm = full_model.intermediate(input, no_activation=True)[-1].norm().item()
            full_init_erank = erank(full_model.intermediate(input, no_activation=True)[-1])
            full_init_mc = mutual_coherence(full_model.intermediate(input, no_activation=True)[-1])

            print('initial norm', init_norm)
            print('initial erank', init_erank)
            print('initial mutual coherence', init_mc)
            print('full initial norm', full_init_norm)
            print('full initial erank', full_init_erank)
            print('full initial mutual coherence', full_init_mc)

        for step in tqdm(range(steps)):
            optimizer.zero_grad()
            output = model(input)

            # if normalized:
            #     final_spectral = model.intermediate(input, no_activation=True)[-1].norm(p=2)
            #     for n, p in model.model.named_parameters():
            #         p.data = p.data / torch.sqrt(final_spectral)

            if algorithm == 'erank':
                l = -erank_tensor(output)
            elif algorithm == 'erank_norm':
                if output.norm() > 1000:
                    l = torch.tensor(0., requires_grad=True)
                else:
                    l = -erank_tensor(output) - features_norm_coeff * output.norm()
            elif algorithm == 'features_frobenius':
                l = -output.norm()
            elif algorithm == 'features_nuclear':
                l = -output.norm(p=1)
            elif algorithm == 'erank_abs':
                l = torch.abs(full_init_erank - erank_tensor(output))
            elif algorithm == 'mutual_coherence':
                l = mutual_coherence_tensor(output)
            elif algorithm == 'srank': # frobenious / spectral
                l = -(output.norm() / output.norm(p=2))
            else:
                raise NotImplementedError
            if weight_reg_coeff < 0:
                for n, p in model.named_parameters():
                    l += torch.norm(p) * weight_reg_coeff
            l.backward()
            optimizer.step()
        if kaiming_balance:
            model._apply_mask()
            for n, p in model.named_parameters():
                if n[6:] in model.prunable_layer_names or n in model.prunable_layer_names:
                    p.data = ((torch.sqrt(torch.tensor(2.)/ p.size(1))) * (1 / p.data[p.data.abs() > 0].std())) * p.data

        with torch.no_grad():
            final_norm = model.intermediate(input, no_activation=True)[-1].norm().item()
            final_erank = erank(model.intermediate(input, no_activation=True)[-1])
            final_mc = mutual_coherence(model.intermediate(input, no_activation=True)[-1])
        print('final norm', final_norm)
        print('final erank', final_erank)
        print('final mutual coherence', final_mc)

        # if normalized:
        #     final_spectral = model.intermediate(input, no_activation=True)[-1].norm(p=2)
        #     for n, p in model.model.named_parameters():
        #         p.data = p.data / torch.sqrt(final_spectral)

        # data_dist = pd.concat([
        #     pd.concat([pd.DataFrame(
        #         {'weight': param.detach().cpu().numpy()[param.abs().cpu().detach().numpy() > 0], 'layer': name[6:]})]
        #      , ignore_index=True)
        # for name, param in model.named_parameters() if name[6:] in model.prunable_layer_names], ignore_index=True)
        #
        # f = sns.histplot(data=data_dist, hue='layer', x='weight', stat="probability", common_norm=False)
        # f.figure.savefig(os.path.join(self.branch_root, 'learned_distribution.pdf'))

        if not get_platform().is_primary_process: return
        if not get_platform().exists(self.branch_root): get_platform().makedirs(self.branch_root)
        with open(os.path.join(self.branch_root, 'properties.json'), 'w') as f:
            json.dump({'init_norm': init_norm, 'init_erank': init_erank, 'init_mutual_coherence': init_mc,
                       'final_norm': final_norm, 'final_erank': final_erank, 'final_mutual_coherence': final_mc,
                       'init_norm_full': full_init_norm, 'init_erank_full': full_init_erank, 'init_mutual_coherence_full': full_init_mc,
                       }, f)

        train.standard_train(model, self.branch_root, self.lottery_desc.dataset_hparams,
                             self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Train initial weights with random masks"

    @staticmethod
    def name():
        return 'train_init'
