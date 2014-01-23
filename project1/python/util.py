import numpy as np


def process_line(line):
    line = line.strip()
    elts = line.split(' ')
    label = int(elts[0])
    pairs = list(e.split(':') for e in elts[1:])
    idx, vals = zip(*pairs)
    sample = np.array(map(int, vals))
    return sample, label


def read_file(filename):
    """Read a data file and return data matrix and labels"""
    f = open(filename)
    lines = f.read().split('\n')
    pairs = list(process_line(i) for i in lines if len(i.strip()) > 0)
    f.close()
    samples, labels = zip(*pairs)
    samples = np.array(samples)
    labels = np.array(labels)
    return samples, labels
