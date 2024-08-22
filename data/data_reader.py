import pandas as pd
import numpy as np

data = pd.read_csv("data/og_data/train.csv")
idx = np.arange(len(data))
np.random.shuffle(idx)
vsize = int(0.2 * len(data))
vidx = np.sort(idx[:vsize])
tidx = np.sort(idx[vsize:])

tdata = data.iloc[tidx]
vdata = data.iloc[vidx]
vdata.to_csv('data/valid.csv')
tdata.to_csv('data/train.csv')
