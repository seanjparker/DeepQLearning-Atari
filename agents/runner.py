import argparse

from dqn.train import train as dqn_train
from dueling_dqn.train import train as ddqn_train

# Create an argument parser for choosing the architecture of the network to train
parser = argparse.ArgumentParser(description='Train/Test RL Algorithms')
parser.add_argument('--algo', default='dqn', choices=['dqn', 'ddqn'], help='Algorithm used for training')
parser.add_argument('--doubledqn', default=False, type=bool, help='Specifiy is double dqn is to be used for training')

# Parse the provided command line arguments
args = parser.parse_args()

# Run the specified training algorithm
print('Training with {}, using DoubleDQN={}'.format(args.algo, args.doubledqn))
if args.algo == 'dqn':
    dqn_train(use_double_dqn=args.doubledqn)
elif args.algo == 'ddqn':
    ddqn_train(use_double_dqn=args.doubledqn)
else:
    print('Not recognised training algorithm')



