import argparse
import yaml
from open_lth import main
import sys


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='myyaml.yaml', type=str, nargs='+')
    args = parser.parse_args()
    for config in args.config:
        with open(config) as file:
            yaml_args = yaml.safe_load(file)
        cmd = [sys.argv[0]]
        if 'subcommand' in yaml_args.keys():
            cmd.append(yaml_args['subcommand'])
            if 'branch_name' in yaml_args.keys() and yaml_args['branch_name']:
                cmd.append(yaml_args['branch_name'])

        for k, v in yaml_args.items():
            if k != 'branch_name' and k != 'subcommand' and v is not None:
                if type(v) is bool and v:
                    cmd.append(f'--{k}')
                    continue
                elif type(v) is bool:
                    continue
                cmd.append(f'--{k}')
                cmd.append(f'{v}')
        print('running:', cmd)
        # args = argparse.Namespace(**yaml_args)
        sys.argv = cmd
        # main(args, args.subcommand)
        main()
