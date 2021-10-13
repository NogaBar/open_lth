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
        subdir = 'lottery_branch_cross_properties_815d159d2bccd309396396148c63d25d'
    if args.model == 'fc_normalized':
        dir_name = '/home/noga/open_lth_data/lottery_efa55fed9865fe839a00d27df3a9969b/replicate_5/'
        subdir = 'lottery_branch_cross_properties_0557bf0ae019d966cbb841f694145244'
    sparsity = []

    data = pd.DataFrame()

    for dir in os.listdir(dir_name):  # level directories
        # if not 'level' in dir or int(dir.split('_')[1]) < 15:
        if not 'level' in dir or int(dir.split('_')[1]) < 15:
            continue
        with open(os.path.join(dir_name, dir, 'main', 'sparsity_report.json'), 'rb') as f_sparse:
            sparse_dict = json.load(f_sparse)
            sparsity.append((sparse_dict['unpruned'] / sparse_dict['total']) * 100)

        level_data = pd.read_csv(os.path.join(dir_name, dir, subdir, 'cross_properties.csv'), index_col=0)
        level_data['sparsity'] = sparsity[-1]
        data = data.append(level_data, ignore_index=True)



    # data = pd.DataFrame({'sparsity': sparsity, 'accuracy': accuracy, 'type': [type] * len(sparsity)})
    data = data[data.sparsity < args.max_sparsity]

    for property in ['erank', 'mutual coherence', 'frobenius', 'min svd']:
        f = sns.lineplot(data=data.loc[(data['property'] == property) & (data['type'] != 'nuclear') & (data['type']!= 'stable rank')], x='sparsity', y='value',
                         hue='type', markers=True, dashes=False, style="type")
        # f = sns.relplot(x='sparsity', y='', kind='line', data=data, hue='type', legend=True, linewidth = 1)
        if property == 'min svd':
            f.set(yscale='log')
        f.set_xlabel('Remaining Weights [%]')
        f.set_ylabel(property)
        f.get_figure().savefig(os.path.join(dir_name, 'plots', f'properties_cross_{args.max_sparsity}_{property}_{args.model}.pdf'))
        f.get_figure().clear()



if __name__ == '__main__':
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='fc', help="model")
    parser.add_argument("--max_sparsity", type=int, default=20, help="max sparsity in plot")
    # parser.add_argument("--property", type=str, default='weights', help="use weights or features properties")
    # parser.add_argument("--es", type=int, default=10, help="early stopping")
    # parser.add_argument("--epochs", type=int, default=40, help="epochs")
    args = parser.parse_args()

    main(args)