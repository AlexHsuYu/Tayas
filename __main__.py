from .cli import NN_test, NN_train, RNN_train, RNN_test
from .tayas import generate_index, PROJECT_ROOT
import pandas as pd, os


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='IMBD2019_final')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")    
    args = parser.parse_args()

    if args.command == "train":
        NN_train()
        RNN_train()
    elif args.command == "detect":
        nn_test = NN_test()
        nn_test = NN_test()
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
    

