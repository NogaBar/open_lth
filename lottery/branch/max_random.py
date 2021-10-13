# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from lottery.branch import base
import models.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train
from utils.tensor_utils import generate_mask_active, erank, shuffle_tensor, mutual_coherence,\
    gradient_mean, activation_mean, activation
import datasets.registry
import numpy as np
import copy
from platforms.platform import get_platform
from foundations import paths
from tqdm import tqdm

erank_frobenius_coeff = 0.75

class Branch(base.Branch):
    def branch_function(self, seed: int, property: str, trials: int, top: int = 1):
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
        if property == 'activation' or property == 'gradient_mean' or property == 'loss' or property == 'accuracy' or property == 'labels_entropy':
            input, labels = list(train_loader)[0]
            input, labels = input.cuda(), labels.cuda()

        # features = orig_model.intermediate(input)
        # features.insert(0, input)
        properties = np.zeros((len(tensors), trials))

        for i, (name, param) in enumerate(tensors.items()):
            curr_mask = Mask()
            for t in tqdm(range(trials)):
                curr_mask[name] = shuffle_tensor(orig_mask[name], seed=seed+t).int()
                if property == 'weight_erank':
                    properties[i, t] = erank(param * curr_mask[name])
                if property == 'weight_min_singular':
                    s = torch.linalg.svdvals(param * curr_mask[name])
                    properties[i, t] = s[-1]
                elif property == 'weight_mutual_coherence':
                    properties[i, t] = -mutual_coherence(param * curr_mask[name])
                elif property == 'weight_frobenius':
                    properties[i, t] = (param * curr_mask[name]).norm().item()
                elif property == 'weight_kaiming':
                    kaiming_std = torch.sqrt(torch.tensor(2.) / param.size(1)) # for fc only!
                    properties[i, t] = -torch.abs(param[curr_mask[name] > 0].std() - kaiming_std)
                elif property == 'features_erank':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    # model[name].data = param * curr_mask[name]
                    features_for_prop = model.intermediate(input)
                    properties[i, t] = erank(features_for_prop[i])
                elif property == 'features_erank_frobenius':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    # model[name].data = param * curr_mask[name]
                    with torch.no_grad():
                        features_for_prop = model.intermediate(input)
                    properties[i, t] = erank_frobenius_coeff * erank(features_for_prop[i]) +\
                                       (1- erank_frobenius_coeff) * features_for_prop[i].norm().item()
                elif property == 'features_mutual_coherence':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    # model[name].data = param * curr_mask[name]
                    features_for_prop = model.intermediate(input)
                    properties[i, t] = -mutual_coherence(features_for_prop[i])
                elif property == 'features_frobenius':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    # model[name].data = param * curr_mask[name]
                    features_for_prop = model.intermediate(input)
                    properties[i, t] = features_for_prop[i].norm().item()
                elif property == 'features_stable_rank':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    # model[name].data = param * curr_mask[name]
                    features_for_prop = model.intermediate(input)
                    properties[i, t] = (torch.linalg.norm(features_for_prop[-1], ord='fro') /
                                        torch.linalg.norm(features_for_prop[-1],ord=2)).item()
                elif property == 'features_nuclear':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    # model[name].data = param * curr_mask[name]
                    features_for_prop = model.intermediate(input)
                    properties[i, t] = features_for_prop[i].norm(p=1).item()
                elif property == 'min_singular':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    # model[name].data = param * curr_mask[name]
                    features_for_prop = model.intermediate(input)
                    _, s, _ = torch.svd(features_for_prop[i], compute_uv=False)
                    properties[i, t] = s[-1]
                elif property == 'activation':
                    if i < len(tensors) - 1:
                        model = copy.deepcopy(model_graduate)
                        model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                        model.cuda()
                        properties[i, t] = activation(model, input)[i]
                    else:
                        properties[i, t] = 0
                elif property == 'gradient_mean':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    model.cuda()
                    properties[i, t] = gradient_mean(model, input, labels)[i]
                elif property == 'loss':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    model.cuda()
                    loss = model.loss_criterion(model(input), labels)
                    properties[i, t] = -loss
                elif property == 'accuracy':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    model.cuda()
                    output = model(input)
                    _, predicted = torch.max(output.data, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    properties[i, t] = correct / total
                elif property == 'labels_entropy':
                    model = copy.deepcopy(model_graduate)
                    model.register_buffer(PrunedModel.to_mask_name(name), curr_mask[name].float())
                    model.cuda()
                    output = model(input)
                    _, predicted = torch.max(output.data, 1)
                    total = labels.size(0)
                    p = predicted.histc(bins=10) / total
                    properties[i, t] = -torch.sum(p[p > 0] * p[p > 0].log()).item()
            best_mask[name] = shuffle_tensor(orig_mask[name], int(seed + np.argmax(properties[i, :]))).int()
            model_graduate.register_buffer(PrunedModel.to_mask_name(name), best_mask[name].float())

        # Save the properties.
        with open(paths.properties(self.branch_root, property), 'wb') as f:
            np.save(f, properties)

        for t in range(1,top+1): # bug fixed, one of them is chosen with the worst property value
            curr_model = copy.deepcopy(base_model)
            best_mask[name] = shuffle_tensor(orig_mask[name], int(seed + np.argsort(properties[i, :])[-t])).int()

            best_mask.save(f'{self.branch_root}_top_{t}')
            if not get_platform().exists(f'{self.branch_root}_top_{t}'): get_platform().makedirs(f'{self.branch_root}_top_{t}')

            # Train the model with the new mask.
            pruned_model = PrunedModel(curr_model, best_mask)
            train.standard_train(pruned_model, f'{self.branch_root}_top_{t}', self.lottery_desc.dataset_hparams,
                                 self.lottery_desc.training_hparams, start_step=start_step, verbose=self.verbose)

    @staticmethod
    def description():
        return "Randomly active prune the model with max property."

    @staticmethod
    def name():
        return 'max_random'
