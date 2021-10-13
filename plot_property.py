import argparse
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from platforms.platform import get_platform
import seaborn as sns


def main(args):
    property= {}
    rand_property = {}
    sparsity = []
    for dir in os.listdir(args.dir):  # level directories
        if not 'level' in dir:
            continue
        with open(os.path.join(args.dir, dir, 'main', 'sparsity_report.json'), 'rb') as f_sparse:
            sparse_dict = json.load(f_sparse)
            sparsity.append((sparse_dict['unpruned'] / sparse_dict['total']) * 100)
        for subdir in os.listdir(os.path.join(args.dir, dir)):  # find the right directory in level
            if 'properties' in subdir:
                path = os.path.join(args.dir, dir, subdir, f'properties_{args.property}.log')
                if args.property in path and args.sub_dir in path:
                    with open(path, 'rb') as f:
                        dict = json.load(f)
                        property[int(dir.split('_')[1])] = dict['lth']
                        rand_property[int(dir.split('_')[1])] = dict['random']

    sparsity = np.array(sparsity)
    keys = np.array(list(rand_property.keys()))
    id = np.argsort(np.array(list(rand_property.keys())))
    colors_list = ['blue', 'green', 'magenta', 'red', 'brown', 'cyan', 'purple', 'grey', 'orange', 'pink', 'lime']
    generator = rand_property[0].keys() if 'weight' in args.property else range(len(rand_property[0]))
    layers = [range(args.layers[0])] if len(args.layers) == 1 else args.layers
    for i, layer in enumerate(generator):
        # if args.layers > 0 and i >= args.layers:
        #     break
        # elif args.layers < 0 and i < len(generator) + args.layers:
        #     continue
        if (i + 1) not in layers:
            continue
        prop_l = np.array([v[layer] for k, v in property.items()])
        prop_rand_l = np.array([v[layer] for k, v in rand_property.items()])
        plt.plot(sparsity[id], prop_l[id], 'o', label=f'lth layer{i+1}', color=colors_list[i])
        plt.plot(sparsity[id], prop_rand_l[id], 'x', label=f'random layer{i+1}', color=colors_list[i])
    plt.yscale('log')
    plt.legend()

    # save figure
    save_dir = os.path.join(args.dir, 'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'property_{args.property}_{args.sub_dir}.pdf'))


if __name__ == '__main__':
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory")
    parser.add_argument("--property", type=str, help="property")
    parser.add_argument("--sub_dir", type=str, help="property directory number")
    parser.add_argument("--layers", nargs='+', type=int, help="number of layers to plot")
    args = parser.parse_args()

    main(args)