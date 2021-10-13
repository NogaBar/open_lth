import argparse
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import csv
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")


def get_best_acc_from_logger(f):
    reader = csv.reader(f)
    max_acc = 0.
    for row in reader:
        if 'accuracy' in row[0]:
            if max_acc < float(row[-1]):
                max_acc = float(row[-1])
    return max_acc


def main(args):
    if args.model == 'fc_fashionmnist':
        dir_name = '/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_5/'
        subdir_name = ['main',
                       'lottery_branch_cross_domain_bda4431f8b45796a42f1ba48201a837b',
                       'lottery_branch_cross_domain_d878f350dd8bb76fe79e48fa5ba8229c',
                       'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f'
                       ]
        types = ['lt', 'kmnist lt', 'fake lt', 'random']
    elif args.model == 'fc_fashionmnist_norm':
        dir_name = '/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_5/'
        subdir_name = ['main',
                       ['/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_6/', 'main'],
                       # 'lottery_branch_normalized_init_a5dfacb8e9e5ba6e5202c2b561ee302f',
                       # 'lottery_branch_max_random_a34c06230163fb6cdbb809481966b905_top_1',
                       # 'lottery_branch_normalized_init_6c7f9bc0dca11ad50a94923ae8eeeec0',
                       'lottery_branch_train_init_792b3e0cb8ffc02eaedf341c5505c9fb',
                       'lottery_branch_train_init_3e6cbc002a9d33cdeb5ee51fbddfe892',
                       'lottery_branch_train_init_5248111555f181abe36cb90436bca85a',
                       'lottery_branch_train_init_710c8ee080049acee6d3a5d658d4a784',
                       'lottery_branch_max_random_a34c06230163fb6cdbb809481966b905_top_1',
                       'lottery_branch_max_random_a34c06230163fb6cdbb809481966b905_top_2',
                       # 'lottery_branch_train_init_40bf9a77427039a8735f5836a9e1e455',
                       # 'lottery_branch_train_init_2e1b286abffdd1acff66613c8c16f6b5',
                       # 'lottery_branch_train_init_d4bc50a59535dd4d9efa6af08d9faf3a',
                       # 'lottery_branch_max_random_bff9fc06233d708656ec70ad2ee7c845_top_2',
                       # 'lottery_branch_active_random_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                       # 'lottery_branch_max_random_91f517c14e3a743b2a3993d12b8fdc35_top_1',
                       # 'lottery_branch_normalized_init_ad573b01a6504fe5a48edef5f4d80fd8',
                       # 'lottery_branch_randomly_reinitialize_7b99794732481fff503d5290ee43fb4b',
                       'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                       'lottery_branch_randomly_prune_7570b2024d9c743dc0727e78e8fc5970'
                       ]
        types = ['lt', 'lt',
                 # 'lt normalized',
                 # 'erank',
                 # 'erank normalized',
                 'pre-train erank',
                 'pre-train erank norm',
                 'pre-train erank norm',
                 'pre-train erank norm',
                 'max erank from random',
                 'max erank from random',
                 # 'train erank from max',
                 # 'train erank lth dist',
                 # 'train nuclear',
                 # 'max nuclear',
                 # 'active',
                 # 'frobenius',
                 # 'frobenius normalized',
                 # 'reinit',
                 'random',
                 'random']
    elif args.model == 'fc_fashionmnist_pretrain':
        dir_name = '/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_5/'
        subdir_name = ['main',
                       'lottery_branch_train_init_792b3e0cb8ffc02eaedf341c5505c9fb',
                       'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f'
                       ]
        types = ['lt',
                 'pre-train',
                 'random']
    elif args.model == 'fc_fashionmnist_simple':
        dir_name = '/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_5/'
        subdir_name = ['main',
                       ['/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_6/', 'main'],
                       'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                       'lottery_branch_randomly_prune_7570b2024d9c743dc0727e78e8fc5970'
                       ]
        types = ['lt',
                 'lt',
                 'random',
                 'random']
    elif args.model == 'fc_kmnist':
        dir_name = '/home/noga/open_lth_data/lottery_262d95790f20ea2d3f35a8c9845560f2/replicate_5'
        subdir_name = ['main',
                       'lottery_branch_cross_domain_5f4dae5531a4c664d79a7502821bf22c',
                       'lottery_branch_cross_domain_d878f350dd8bb76fe79e48fa5ba8229c',
                       'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f'
                       ]
        types = ['lt', 'mnist lt', 'fake lt', 'random']
    elif args.model == 'conv6_cifar10':
        dir_name = '/home/noga/open_lth_data/lottery_9ae2c18b35685fc44452965ff04242fc/replicate_5/'
        subdir_name = ['main',
                       ['/home/noga/open_lth_data/lottery_9ae2c18b35685fc44452965ff04242fc/replicate_6/', 'main'],
                       'lottery_branch_randomly_prune_34e871f7d18325fd7be231498e0e51cb',
                       'lottery_branch_randomly_prune_7570b2024d9c743dc0727e78e8fc5970',
                       # 'lottery_branch_cross_domain_db1ab0a9e28e21ec930328aab0c77607',
                       # 'lottery_branch_cross_domain_c0df7ff93020953346ca8571388c325e',,
                       # 'lottery_branch_randomly_reinitialize_7b99794732481fff503d5290ee43fb4b',
                       'lottery_branch_train_init_2e2b2efb071e4e8c2179557d120ae515',
                       'lottery_branch_train_init_d755a779acd852ea3acd6afe21c15c56',
                       'lottery_branch_train_init_57bb2c89b957babc0d157da7aa49f12b',
                       'lottery_branch_train_init_e830aaab46170af40a42a05f3c346ecd',
                       ]
        types = ['lt', 'lt',
                 # 'cifar100 lt',
                 # 'random lt',
                 'random',
                 'random',
                 # 'reinit',
                 'pre-train erank',
                 'pre-train erank',
                 'pre-train erank norm',
                 'pre-train erank norm',
                 ]
    elif args.model == 'conv6_cifar10_simple':
        dir_name = '/home/noga/open_lth_data/lottery_9ae2c18b35685fc44452965ff04242fc/replicate_5/'
        subdir_name = ['main',
                       ['/home/noga/open_lth_data/lottery_9ae2c18b35685fc44452965ff04242fc/replicate_6/', 'main'],
                       'lottery_branch_randomly_prune_34e871f7d18325fd7be231498e0e51cb',
                       'lottery_branch_randomly_prune_7570b2024d9c743dc0727e78e8fc5970',
                       ]
        types = ['lt', 'lt',
                 'random',
                 'random',
                 ]
    elif args.model == 'conv2_cifar10_simple':
        dir_name = '/home/noga/open_lth_data/lottery_f3aa776875c9f115675a16cdf7dfa7d3/replicate_5/'
        subdir_name = ['main',
                       'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                       ]
        types = ['lt',
                 'random',
                 ]
    elif args.model == 'conv6_cifar100_cross':
        dir_name = '/home/noga/open_lth_data/lottery_8d69a64239fb93f73edeb70953a2235a/replicate_5/'
        subdir_name = ['main',
                       'lottery_branch_cross_domain_5e5ad05bcd277e33a4aa1edb9459a237',
                       'lottery_branch_cross_domain_c0df7ff93020953346ca8571388c325e',
                       'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                       ]
        types = ['lt',
                 'cifar10 lt',
                 'random lt',
                 'random',
                 ]
    elif args.model == 'vgg19':
        dir_name = '/home/noga/open_lth_data/lottery_5d6a1d49157fd9e7a5d2bd6b18da352e/replicate_5/'
        subdir_name = ['main',
                       # 'lottery_branch_cross_domain_db1ab0a9e28e21ec930328aab0c77607',
                       # 'lottery_branch_cross_domain_c0df7ff93020953346ca8571388c325e',
                       # 'lottery_branch_randomly_prune_a5dfacb8e9e5ba6e5202c2b561ee302f',
                       # 'lottery_branch_randomly_reinitialize_7b99794732481fff503d5290ee43fb4b',
                       # 'lottery_branch_train_init_a5dfacb8e9e5ba6e5202c2b561ee302f',
                       # 'lottery_branch_train_init_3e5248a22383317ef1e509dc24f1ce10',
                       ]
        types = ['lt',
                 # 'cifar100 lt',
                 # 'random lt',
                 # 'random',
                 # 'reinit',
                 # 'train erank'
                 ]
    sparsity = []
    data = pd.DataFrame({'sparsity': [], 'accuracy': [], 'type': []})

    for dir in os.listdir(dir_name):  # level directories
        # if not 'level' in dir:
        # if not 'level' in dir or int(dir.split('_')[1]) < 15:
        if not 'level' in dir or int(dir.split('_')[1]) < args.min_level:
            continue
        with open(os.path.join(dir_name, dir, subdir_name[0], 'sparsity_report.json'), 'rb') as f_sparse:
            sparse_dict = json.load(f_sparse)
            sparsity.append((sparse_dict['unpruned'] / sparse_dict['total']) * 100)

        for subdir, type in zip(subdir_name, types):
            if isinstance(subdir, list):
                path = os.path.join(subdir[0], dir, subdir[1], 'logger')
            else:
                path = os.path.join(dir_name, dir, subdir, 'logger')
            with open(path) as f:
                accuracy = get_best_acc_from_logger(f)
            data = data.append({'sparsity': sparsity[-1], 'accuracy': accuracy, 'type': type}, ignore_index=True)

    # data = pd.DataFrame({'sparsity': sparsity, 'accuracy': accuracy, 'type': [type] * len(sparsity)})
    data = data[data.sparsity < args.max_sparsity]
    # f = sns.relplot(x='sparsity', y='accuracy', kind='line', data=data, hue='type', legend=True)
    f = sns.lineplot(x='sparsity', y='accuracy', data=data, hue='type', legend=True)
    f.set_xlabel('Remaining Weights [%]')
    f.get_figure().set_size_inches(5, 5)
    # f.set(title='Conv2 Accuracy')
    # f.fig.savefig(os.path.join(dir_name, 'plots', f'accuracy_cross_{args.max_sparsity}_{args.model}.pdf'))
    f.get_figure().savefig(os.path.join(dir_name, 'plots', f'accuracy_cross_{args.max_sparsity}_{args.model}.pdf'))





if __name__ == '__main__':
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='fc_mnist',help="model")
    parser.add_argument("--max_sparsity", type=int, default=100, help="max sparsity")
    parser.add_argument("--min_level", type=int, default=100, help="max sparsity")
    # parser.add_argument("--experiments", type=str, nargs='+', help="experiments to present")
    args = parser.parse_args()

    main(args)