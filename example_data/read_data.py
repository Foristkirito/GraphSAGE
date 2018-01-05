import numpy as np
from tempfile import mkdtemp
filename = "ppi-feats.npy"

fp = np.memmap(filename, dtype='float64', offset = 50 * 8, mode='r', shape=(14754,50))
# print fp.shape