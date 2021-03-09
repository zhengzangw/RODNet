import argparse
import pickle

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("--seq")
    parser.add_argument("--num", type=int)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    confmaps = pickle.load(open(args.input, "rb"))
    confmap = confmaps[args.seq][args.num]
    confmap = np.transpose(confmaps, (1, 2, 0))
    
    plt.imshow(confmapc[class_id], origin="lower", aspect="auto")
