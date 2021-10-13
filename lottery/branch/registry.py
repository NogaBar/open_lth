# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from lottery.branch.base import Branch
from lottery.branch import randomly_prune
from lottery.branch import randomly_reinitialize
from lottery.branch import retrain
from lottery.branch import properties
from lottery.branch import active_random_prune
from lottery.branch import max_active_prune
from lottery.branch import property_distribution
from lottery.branch import cross_domain
from lottery.branch import train_init
from lottery.branch import max_random
from lottery.branch import singular_values
from lottery.branch import normalized_init
from lottery.branch import weight_distribution
from lottery.branch import cross_properties
from lottery.branch import prune_with_erank

registered_branches = {
    'randomly_prune': randomly_prune.Branch,
    'randomly_reinitialize': randomly_reinitialize.Branch,
    'retrain': retrain.Branch,
    'properties': properties.Branch,
    'active_random_prune': active_random_prune.Branch,
    'max_active_prune': max_active_prune.Branch,
    'property_distribution': property_distribution.Branch,
    'cross_domain': cross_domain.Branch,
    'train_init': train_init.Branch,
    'max_random': max_random.Branch,
    'singular_values': singular_values.Branch,
    'normalized_init': normalized_init.Branch,
    'weight_distribution': weight_distribution.Branch,
    'cross_properties': cross_properties.Branch,
    'prune_with_erank': prune_with_erank.Branch,
}


def get(branch_name: str) -> Branch:
    if branch_name not in registered_branches:
        raise ValueError('No such branch: {}'.format(branch_name))
    else:
        return registered_branches[branch_name]
