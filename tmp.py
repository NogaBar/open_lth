import yaml

d = {'subcommand': 'lottery', 'platform': 'local', 'display_output_location': False, 'num_workers': 0, 'gpu': '6',
     'replicate': 2, 'default_hparams': 'mnist_lenet_300_100', 'quiet': False, 'evaluate_only_at_end': False,
     'levels': 0, 'rewinding_steps': None, 'pretrain': False, 'dataset_name': 'fashionmnist', 'batch_size': 128,
     'do_not_augment': False, 'transformation_seed': None, 'subsample_fraction': None, 'random_labels_fraction': None,
     'unsupervised_labels': None, 'blur_factor': None, 'model_name': 'mnist_lenet_300_100',
     'model_init': 'kaiming_normal', 'batchnorm_init': 'uniform', 'batchnorm_frozen': False, 'output_frozen': False,
     'others_frozen': False, 'others_frozen_exceptions': None, 'optimizer_name': 'sgd', 'lr': 0.1,
     'training_steps': '40ep', 'data_order_seed': None, 'momentum': 0.0, 'nesterov_momentum': 0.0,
     'milestone_steps': None, 'gamma': None, 'warmup_steps': None, 'weight_decay': None, 'apex_fp16': False,
     'pruning_strategy': 'sparse_global', 'pruning_fraction': 0.2, 'pruning_layers_to_ignore': 'fc.weight'}

with open(r'./myyaml.yaml', 'w') as file:
    print(yaml.dump(d, file))
