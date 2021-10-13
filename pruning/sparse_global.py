# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import numpy as np

from foundations import hparams
import models.base
from pruning import base
from pruning.mask import Mask


@dataclasses.dataclass
class PruningHparams(hparams.PruningHparams):
    pruning_fraction: float = 0.2
    pruning_layers_to_ignore: str = None

    _name = 'Hyperparameters for Sparse Global Pruning'
    _description = 'Hyperparameters that modify the way pruning occurs.'
    _pruning_fraction = 'The fraction of additional weights to prune from the network.'
    _layers_to_ignore = 'A comma-separated list of addititonal tensors that should not be pruned.'


class Strategy(base.Strategy):
    @staticmethod
    def get_pruning_hparams() -> type:
        return PruningHparams

    @staticmethod
    def prune(pruning_hparams: PruningHparams, trained_model: models.base.Model, current_mask: Mask = None):
        current_mask = Mask.ones_like(trained_model).numpy() if current_mask is None else current_mask.numpy()

        # Determine which layers can be pruned.
        prunable_tensors = set(trained_model.prunable_layer_names)
        output_name = list(set(trained_model.output_layer_names) & prunable_tensors)[0]
        prunable_conv = set(trained_model.prunable_conv_names)
        prunable_fc = prunable_tensors - prunable_conv
        prunable_fc.remove(output_name)

        # Determine the number of weights that need to be pruned.
        number_of_remaining_weights_fc = np.sum([np.sum(current_mask[k]) for k in prunable_fc])
        number_of_remaining_weights_conv = np.sum([np.sum(current_mask[k]) for k in prunable_conv])
        number_of_remaining_weights_out = np.sum(current_mask[output_name])

        number_of_weights_to_prune_fc = np.ceil(
            pruning_hparams.pruning_fraction * number_of_remaining_weights_fc).astype(int)
        number_of_weights_to_prune_conv = np.ceil(
            pruning_hparams.pruning_conv * number_of_remaining_weights_conv).astype(int)
        number_of_weights_to_prune_out = np.ceil(
            pruning_hparams.pruning_fraction_last_fc * number_of_remaining_weights_out).astype(int)

        if pruning_hparams.pruning_layers_to_ignore:
            ignored = pruning_hparams.pruning_layers_to_ignore
            prunable_tensors -= set(ignored.split(','))

        # Get the model weights.
        weights_fc = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_fc}
        weights_conv = {k: v.clone().cpu().detach().numpy()
                   for k, v in trained_model.state_dict().items()
                   if k in prunable_conv}
        output_weight = trained_model.state_dict()[output_name].clone().cpu().detach().numpy()

        # Create a vector of all the unpruned weights in the model.
        weight_vector_fc = None if len(weights_fc) == 0 else np.concatenate([v[current_mask[k] == 1] for k, v in weights_fc.items()])
        weight_vector_conv = None if len(weights_conv) == 0 else np.concatenate([v[current_mask[k] == 1] for k, v in weights_conv.items()])
        weight_vector_out = output_weight[current_mask[output_name] == 1]


        threshold_fc = np.sort(np.abs(weight_vector_fc))[number_of_weights_to_prune_fc] if weight_vector_fc is not None else None
        threshold_conv = np.sort(np.abs(weight_vector_conv))[number_of_weights_to_prune_conv] if weight_vector_conv is not None else None
        threshold_out = np.sort(np.abs(weight_vector_out))[number_of_weights_to_prune_out]


        new_mask = Mask()
        for k, v in weights_fc.items():
            new_mask[k] = np.where(np.abs(v) > threshold_fc, current_mask[k], np.zeros_like(v))
        for k, v in weights_conv.items():
            new_mask[k] = np.where(np.abs(v) > threshold_conv, current_mask[k], np.zeros_like(v))

        new_mask[output_name] = np.where(np.abs(output_weight) > threshold_out, current_mask[output_name],
                                         np.zeros_like(output_weight))

        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]

        return new_mask
