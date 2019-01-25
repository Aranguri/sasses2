import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
args = parser.parse_args()
print(args.lr)
