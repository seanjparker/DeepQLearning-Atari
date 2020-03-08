import argparse

from dqn.train import train as dqn_train
from dueling_dqn.train import train as ddqn_train

parser = argparse.ArgumentParser(description='Train/Test RL Algorithms')
parser.add_argument('--algo', default='dqn', help='Algorithm used for training, one of [dqn, ddqn] ')

args = parser.parse_args()

if args.algo == 'dqn':
    dqn_train()
elif args.algo == 'ddqn':
    ddqn_train()
else:
    print('Not recognised algorithm')



