import argparse
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")

import pandas as pd
from platforms.platform import get_platform
import csv


def get_best_acc_from_logger(f, es, epochs):
    reader = csv.reader(f)
    max_acc = 0.
    iters_es = 0
    epoch = 0
    for row in reader:
        if iters_es >= es or epoch == epochs:
            return max_acc
        if 'accuracy' in row[0]:
            epoch += 1
            if max_acc < float(row[-1]):
                max_acc = float(row[-1])
            else:
                iters_es += 1
    return max_acc

def main(args):
    if args.model == 'fc':
        dir_name = '/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_5/'
        if args.property == 'weights':
            subdir_name = ['main',
                           ['/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_6/', 'main'],
                           'lottery_branch_max_random_6280bf44f29a2d812208bd76f07a939e',
                           'lottery_branch_max_random_94967ce6ed693a090b3afbcf7ee88ffa',
                           'lottery_branch_max_active_prune_3571f2a66a14f61ebd3ef3f6b0d98a1e',
                           'lottery_branch_max_random_a8e9401c444c73801bf1ec1bb0d8ca5a',
                           'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                           'lottery_branch_randomly_prune_7570b2024d9c743dc0727e78e8fc5970'
                           ]
            types = ['lt',
                     'lt',
                     'weights erank',
                     'weights mutual coherence',
                     'weights norm',
                     'weights min singular',
                     'random',
                     'random']
            top = 2
        elif args.property == 'features':
            subdir_name = ['main',
                           ['/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_6/', 'main'],
                           'lottery_branch_max_random_a34c06230163fb6cdbb809481966b905',
                           'lottery_branch_max_random_6a77658fc8cc7bea2fe3cc1242717394',
                           'lottery_branch_max_random_91f517c14e3a743b2a3993d12b8fdc35',
                           'lottery_branch_max_random_ba02ceb15f0dbc608eb2a66d867ff5d0',
                           # 'lottery_branch_max_random_6dd9deb0e87d209da19831d47792b54f',
                           # 'lottery_branch_max_random_b607d96bf41f369e264738afe31a455b',
                           # 'lottery_branch_max_random_b7f75ceb565d1133dc89d7aa9a15c9af',
                           # 'lottery_branch_max_random_d03b58f1fa6b48d415d632ed891d0c74',
                           # 'lottery_branch_randomly_reinitialize_7b99794732481fff503d5290ee43fb4b',
                           'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                           'lottery_branch_randomly_prune_7570b2024d9c743dc0727e78e8fc5970'
                           ]
            types = ['lt',
                     'lt',
                     'features erank',
                     'features mutual coherence',
                     'features norm',
                     'features min singular',
                     # 'features stable rank',
                     # 'features gradient mean',
                     # 'features erank norm',
                     # 'accuracy',
                     # 'reinit',
                     'random',
                     'random']
            top = 2

    if args.model == 'fc_maximize':
        dir_name = '/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_5/'

        subdir_name = ['main',
                       'lottery_branch_train_init_cb6392f651c9ee915d2dbbcb5fe656d6',
                       'lottery_branch_train_init_9e8555bdc63457f10d1f6756bb00192c',
                       'lottery_branch_train_init_b4a8615b015afa45b4e20e88a6146158',
                       'lottery_branch_train_init_f170a48501d55fbf674d7b27075cf875',
                       'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f'
                       ]
        types = ['lt',
                 'erank abs',
                 'erank',
                 'norm',
                 'mutual coherence',
                 'random']
        top = 1
    if 'kmnist' in args.model:
        dir_name = '/home/noga/open_lth_data/lottery_262d95790f20ea2d3f35a8c9845560f2/replicate_5'
        if args.property == 'weights':
            subdir_name = ['main',
                           'lottery_branch_max_random_6280bf44f29a2d812208bd76f07a939e',
                           'lottery_branch_max_random_94967ce6ed693a090b3afbcf7ee88ffa',
                           'lottery_branch_max_active_prune_3571f2a66a14f61ebd3ef3f6b0d98a1e',
                           'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                           ]
            types = ['lt',
                     'weights erank',
                     'weights mutual coherence',
                     'weights norm',
                     'random']
            top = 2
        elif args.property == 'features':
            subdir_name = ['main',
                           'lottery_branch_max_random_a34c06230163fb6cdbb809481966b905',
                           'lottery_branch_max_random_6a77658fc8cc7bea2fe3cc1242717394',
                           'lottery_branch_max_random_91f517c14e3a743b2a3993d12b8fdc35',
                           'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                           ]
            types = ['lt',
                     'features erank',
                     'features mutual coherence', 'features norm', 'random']
            top = 2
    sparsity = []

    data = pd.DataFrame({'sparsity': [], 'accuracy': [], 'type': []})

    for dir in os.listdir(dir_name):  # level directories
        # if not 'level' in dir or int(dir.split('_')[1]) < 10:
        if not 'level' in dir:
            continue
        with open(os.path.join(dir_name, dir, subdir_name[0], 'sparsity_report.json'), 'rb') as f_sparse:
            sparse_dict = json.load(f_sparse)
            sparsity.append((sparse_dict['unpruned'] / sparse_dict['total']) * 100)

        for subdir, type in zip(subdir_name, types):
            if 'weights' in type or 'features' in type or 'accuracy' in type:
                curr_top = top
            else:
                curr_top = None
            if isinstance(subdir, list):
                path = os.path.join(subdir[0], dir, subdir[1], 'logger')
            else:
                path = os.path.join(dir_name, dir, subdir, 'logger')
            if curr_top is None:
                with open(path) as f:
                    accuracy = get_best_acc_from_logger(f, args.es, args.epochs)
                data = data.append({'sparsity': sparsity[-1], 'accuracy': accuracy, 'type': type}, ignore_index=True)
            else:
                for t in range(curr_top):
                    path = os.path.join(dir_name, dir, f'{subdir}_top_{t+1}', 'logger')
                    with open(path) as f:
                        accuracy = get_best_acc_from_logger(f, args.es, args.epochs)
                    data = data.append({'sparsity': sparsity[-1], 'accuracy': accuracy, 'type': type}, ignore_index=True)

    # data = pd.DataFrame({'sparsity': sparsity, 'accuracy': accuracy, 'type': [type] * len(sparsity)})
    data = data[data.sparsity < args.max_sparsity]
    f = sns.lineplot(x='sparsity', y='accuracy', data=data, hue='type', legend=True, markers=True, dashes=False, style="type")
    f.set_xlabel('Remaining Weights [%]')

    f.get_figure().savefig(os.path.join(dir_name, 'plots', f'property_corr_spars{args.max_sparsity}_{args.property}.pdf'))



if __name__ == '__main__':
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='fc', help="model")
    parser.add_argument("--max_sparsity", type=int, default=20, help="max sparsity in plot")
    parser.add_argument("--property", type=str, default='features', help="use weights or features properties")
    parser.add_argument("--es", type=int, default=10, help="early stopping")
    parser.add_argument("--epochs", type=int, default=40, help="epochs")
    args = parser.parse_args()

    main(args)