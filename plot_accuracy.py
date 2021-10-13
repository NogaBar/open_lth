import argparse
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import csv
# import seaborn as sns


def main(args):
    sparsity = []
    best_acc_rand = []

    best_acc = []
    sparsity_max_active = []
    sparsity_active = []

    best_max_active = []
    best_acc_active = []

    best_acc_cross = []
    for dir in os.listdir(args.dir):  # level directories
        if not 'level' in dir:
            continue
        with open(os.path.join(args.dir, dir, 'main', 'sparsity_report.json'), 'rb') as f_sparse:
            sparse_dict = json.load(f_sparse)
            sparsity.append((sparse_dict['unpruned'] / sparse_dict['total']) * 100)
        for subdir in os.listdir(os.path.join(args.dir, dir)):  # find the right directory in level
            if 'randomly_prune' in subdir and 'randomly_prune' in args.experiments:
                path = os.path.join(args.dir, dir, subdir, 'logger')
                with open(path) as f:
                    reader = csv.reader(f)
                    max_acc = 0.
                    for row in reader:
                        if 'accuracy' in row[0]:
                            if max_acc < float(row[-1]):
                                max_acc = float(row[-1])
                best_acc_rand.append(max_acc)
            elif 'main' in subdir:
                path = os.path.join(args.dir, dir, subdir, 'logger')
                with open(path) as f:
                    reader = csv.reader(f)
                    max_acc = 0.
                    for row in reader:
                        if 'accuracy' in row[0]:
                            if max_acc < float(row[-1]):
                                max_acc = float(row[-1])
                best_acc.append(max_acc)
            elif 'max_active_prune' in subdir and 'max_active_prune' in args.experiments:
                with open(os.path.join(args.dir, dir, subdir, 'sparsity_report.json'), 'rb') as f_sparse:
                    sparse_dict = json.load(f_sparse)
                    sparsity_max_active.append((sparse_dict['unpruned'] / sparse_dict['total']) * 100)
                path = os.path.join(args.dir, dir, subdir, 'logger')
                with open(path) as f:
                    reader = csv.reader(f)
                    max_acc = 0.
                    for row in reader:
                        if 'accuracy' in row[0]:
                            if max_acc < float(row[-1]):
                                max_acc = float(row[-1])
                best_max_active.append(max_acc)
            elif 'active_random_prune' in subdir and 'active_random_prune' in args.experiments:
                with open(os.path.join(args.dir, dir, subdir, 'sparsity_report.json'), 'rb') as f_sparse:
                    sparse_dict = json.load(f_sparse)
                    sparsity_active.append((sparse_dict['unpruned'] / sparse_dict['total']) * 100)
                path = os.path.join(args.dir, dir, subdir, 'logger')
                with open(path) as f:
                    reader = csv.reader(f)
                    max_acc = 0.
                    for row in reader:
                        if 'accuracy' in row[0]:
                            if max_acc < float(row[-1]):
                                max_acc = float(row[-1])
                best_acc_active.append(max_acc)
            elif 'cross_domain' in subdir and 'cross_domain' in args.experiments:
                path = os.path.join(args.dir, dir, subdir, 'logger')
                with open(path) as f:
                    reader = csv.reader(f)
                    max_acc = 0.
                    for row in reader:
                        if 'accuracy' in row[0]:
                            if max_acc < float(row[-1]):
                                max_acc = float(row[-1])
                best_acc_cross.append(max_acc)

    sparsity = np.array(sparsity)
    sparsity_active = np.array(sparsity_active)
    sparsity_max_active = np.array(sparsity_max_active)
    id = np.argsort(np.array(sparsity))
    colors_list = ['blue', 'green', 'magenta', 'red', 'brown', 'cyan', 'purple', 'grey', 'orange', 'pink', 'lime']

    if 'main' in args.experiments:
        plt.plot(sparsity[id], np.array(best_acc)[id], 'o-', label='lth accuracy', color=colors_list[0])
    if 'randomly_prune' in args.experiments:
        plt.plot(sparsity[id], np.array(best_acc_rand)[id], 'o-', label='random accuracy', color=colors_list[1])
    if 'max_active_prune' in args.experiments:
        plt.plot(sparsity_max_active[id], np.array(best_max_active)[id], 'o-', label='max active accuracy', color=colors_list[2])
    if 'active_random_prune' in args.experiments:
        plt.plot(sparsity_active[id], np.array(best_acc_active)[id], 'o-', label='active accuracy', color=colors_list[3])
    if 'cross_domain' in args.experiments:
        plt.plot(sparsity[id], np.array(best_acc_cross)[id], 'o-', label='cross domain', color=colors_list[4])

    plt.legend()

    # save figure
    save_dir = os.path.join(args.dir, 'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'accuracies.pdf'))


if __name__ == '__main__':
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory")
    parser.add_argument("--experiments", type=str, nargs='+', help="experiments to present")
    args = parser.parse_args()

    main(args)