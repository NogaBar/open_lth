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
        if args.property == 'features_erank_cross':
            subdir = 'lottery_branch_properties_6a411806ffd88e33d7f3ecf53dd93d5e'
        elif args.property == 'features_erank':
            subdir = 'lottery_branch_properties_d7cc31b4c9bac1de2ba41067ff3fe3b4'
        elif args.property == 'features_min_singular':
            subdir = 'lottery_branch_properties_6e8aa46bcba0692c0a3cf1fac1d11e3d'
        elif args.property == 'features_frobenius_cross':
            subdir = 'lottery_branch_properties_8021ee87aa4afc45e9828e914ec20e43'
        elif args.property == 'features_frobenius':
            subdir = 'lottery_branch_properties_a1672ab50757cf60cd13cbcbce8cd5bb'
        elif args.property == 'features_mutual_coherence':
            subdir = 'lottery_branch_properties_b5ebf5e602aff0fd7f9536446dc551ee'
        elif args.property == 'weight_frobenius':
            subdir = 'lottery_branch_properties_f159124d35854a8582bd0f665722e59e'
        elif args.property == 'weight_erank':
            subdir = 'lottery_branch_properties_a5dfacb8e9e5ba6e5202c2b561ee302f'
        elif args.property == 'weight_mutual_coherence':
            subdir = 'lottery_branch_properties_425f3d9430ee5177d6987545632054d0'
        elif args.property == 'weight_min_singular':
            subdir = 'lottery_branch_properties_2145913b819382387afd6937436cee8b'
        elif args.property == 'activation':
            subdir = 'lottery_branch_properties_019a4b315af0dcaaeb2dcc16c3b2929d'
        elif args.property == 'gradients_mean':
            subdir = 'lottery_branch_properties_9061b8af63fd5db0ef93fa5e510dcd55'

    elif args.model == 'conv2':
        dir_name = '/home/noga/open_lth_data/lottery_f3aa776875c9f115675a16cdf7dfa7d3/replicate_5'
        if args.property == 'features_erank':
            subdir = 'lottery_branch_properties_47601468f9297d18c0519c318e78e14e'
            # or lottery_branch_properties_d7cc31b4c9bac1de2ba41067ff3fe3b4 for all 4 last features
        elif args.property == 'features_frobenius':
            # all 6 intemidiate features
            subdir = 'lottery_branch_properties_79b3914a145511cc2578b1f1cfcf283c'
        elif args.property == 'weight_erank':
            subdir = 'lottery_branch_properties_d5e025f36fa9c3572a65097527543996'
        elif args.property == 'weight_frobenius':
            subdir = 'lottery_branch_properties_a6caaa5ad758eae8130b2003e4a1423a'

    elif args.model == 'conv6':
        dir_name = '/home/noga/open_lth_data/lottery_9ae2c18b35685fc44452965ff04242fc/replicate_5'
        if args.property == 'weight_erank':
            subdir = 'lottery_branch_properties_d5e025f36fa9c3572a65097527543996'
        elif args.property == 'weight_frobenius':
            subdir = 'lottery_branch_properties_a6caaa5ad758eae8130b2003e4a1423a'
        elif args.property == 'features_frobenius':
            subdir = 'lottery_branch_properties_79b3914a145511cc2578b1f1cfcf283c'
        elif args.property == 'features_erank':
            subdir = 'lottery_branch_properties_d7cc31b4c9bac1de2ba41067ff3fe3b4'


    sparsity = []

    data = pd.DataFrame()

    for dir in os.listdir(dir_name):  # level directories
        if not 'level' in dir:
            continue
        with open(os.path.join(dir_name, dir, 'main', 'sparsity_report.json'), 'rb') as f_sparse:
            sparse_dict = json.load(f_sparse)
            sparsity.append((sparse_dict['unpruned'] / sparse_dict['total']) * 100)

        level_data = pd.read_json(os.path.join(dir_name, dir, subdir, f'properties_{args.property}.log'), orient='index')
        level_data['type'] = level_data.index

        level_data['index'] = range(len(level_data))
        level_data = level_data.set_index('index')
        level_data['sparsity'] = sparsity[-1]
        level_data['data_type'] = 'natural'
        for i, t in level_data['type'].items():
            if 'rand_data' in t:
                level_data.loc[i, 'data_type'] = 'random'
                level_data.loc[i, 'type'] = t.replace('_rand_data', '')
        # level_data['layer'] = level_data.index
        data = data.append(level_data, ignore_index=True)



    # data = pd.DataFrame({'sparsity': sparsity, 'accuracy': accuracy, 'type': [type] * len(sparsity)})
    data = data[data.sparsity < args.max_sparsity]

    plot_layers = []
    if args.property == 'activation':
        plot_layers.append(0)
    elif 'cross' in args.property:
        plot_layers.append(2)
    else:
        named = False
        for k in data.keys():
            if isinstance(k, str) and 'model' in k:
                named = True
                break
        if named:
            if args.model == 'conv6':
                plot_layers = [
                    'model.layers.0.conv.weight',
                    'model.layers.4.conv.weight',
                    'model.layers.7.conv.weight',
                    'model.fc.weight'
                ]
                new_labels = [
                    'lt conv layer 1', 'random conv layer 1',
                    'lt conv layer 4', 'random conv layer 4',
                    'lt conv layer 6', 'random conv layer 6',
                    'lt last fc layer', 'random last fc layer',
                ]
            else:
                plot_layers = [k for k in data.keys() if 'model.layers.1.conv' in k or 'fc' in k]
                new_labels = [
                    'lt conv layer 2', 'random conv layer 2',
                    'lt fc layer 1', 'random fc layer 1',
                    'lt fc layer 2', 'random fc layer 2',
                    'lt fc layer 3', 'random fc layer 3',
                ]
        else:
            if args.model == 'conv2':
                plot_layers = [2, 3, 4, 5]
                new_labels = [
                    'lt conv layer 2',  'random conv layer 2',
                    'lt fc layer 1',  'random fc layer 1',
                    'lt fc layer 2',  'random fc layer 2',
                    'lt fc layer 3',  'random fc layer 3',
                ]
            elif args.model == 'conv6':
                plot_layers = [0, 1, 2, 3]
                new_labels = [
                    'lt conv layer 1', 'random conv layer 1',
                    'lt conv layer 4', 'random conv layer 4',
                    'lt conv layer 6', 'random conv layer 6',
                    'lt last fc layer', 'random last fc layer',
                ]
            else:
                plot_layers = [0, 1, 2]
                new_labels = []
                for l in plot_layers:
                    new_labels.extend([f'lt layer {l + 1}', f'random layer {l + 1}'])

    for i, l in enumerate(plot_layers):
        if len(set(data['data_type'])) == 1:
            f = sns.lineplot(data=data, x='sparsity', y=l, hue='type', markers=True, dashes=False, style="type",
                             palette=[f'C{i}']*2, legend=True)
            # linestyle=''
            # f.legend().remove()
            f.get_figure().legend().remove()
            # replace labels

            # for t, x in zip(f.get_figure().legend().texts, new_labels): t.set_text(x)
            for t, x in zip(f.legend().texts, new_labels): t.set_text(x)
        else:
            f = sns.lineplot(data=data, x='sparsity', y=l, hue='type', style='data_type')
    f.get_figure().legend().remove()
    f.get_figure().set_size_inches(5, 5)
    # f.legend(loc='best')
    f.set_ylabel(f'{args.property.replace("_", " ")}')
    f.set_yscale('log')
    # f.get_figure().axes[0].locator_params(axis='x', numticks=25)
    # formatter = matplotlib.ticker.ScalarFormatter(useLocale=True)
    # formatter = formatter.set_locs(np.logspace(min(sparsity), max(sparsity), 25))
    # f.xaxis.set_major_formatter(formatter)
    # f.xaxis.locator_params(axis='x', numticks=25)
    f.set_xlabel(f'Remaining Weights [%]')
    f.get_figure().savefig(os.path.join(dir_name, 'plots', f'properties_{args.max_sparsity}_{args.property}_sns.pdf'))
    f.get_figure().clear()



if __name__ == '__main__':
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='fc', help="model")
    parser.add_argument("--max_sparsity", type=int, default=100, help="max sparsity in plot")
    parser.add_argument("--property", type=str, default='features_erank', help="property to plot")
    # parser.add_argument("--property", type=str, default='weights', help="use weights or features properties")
    # parser.add_argument("--es", type=int, default=10, help="early stopping")
    # parser.add_argument("--epochs", type=int, default=40, help="epochs")
    args = parser.parse_args()

    main(args)